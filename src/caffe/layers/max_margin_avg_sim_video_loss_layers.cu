
#include "caffe/l_loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_get_centroid(const Dtype* bottom_data, Dtype* out,
                               const int clip_size, const int batch_size, const int feature_size) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < batch_size * feature_size) {
    int i = thread_id / feature_size, d = thread_id % feature_size;
    Dtype feature_val = 0;
    for (int t = 0; t < clip_size; ++ t) {
      int idx = (t * batch_size + i) * feature_size + d;
      feature_val += bottom_data[idx] / clip_size;
    }
    out[i * feature_size + d] = feature_val;
  }
}

template <typename Dtype>
__global__ void kernel_label_indicator(const Dtype* label, Dtype* out, const int batch_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size * batch_size) {
    int i = id / batch_size, j = id % batch_size;
    int label1 = (int) label[i], label2 = (int) label[j];
    bool has_common = (label1 & label2) > 0;
    out[id] = has_common ? 1 : -1;
  }
}

template <typename Dtype>
__global__ void kernel_pairwise_loss(const Dtype* S_video, const Dtype* I_video, Dtype* out, 
                                     const int count, const Dtype margin_width, const Dtype bias) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < count) {
    Dtype pre_loss = margin_width - I_video[id] * (S_video[id] + bias);
    out[id] = pre_loss > 0 ? pre_loss : 0;
  }
}    

template <typename Dtype>
void MaxMarginAvgSimVideoLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();

  // calculate video centroid
  Dtype* video_centroid = video_centroid_.mutable_gpu_data();
  kernel_get_centroid<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * feature_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, video_centroid,
                              clip_size_, batch_size_, feature_size_);
  
  // calculate video pair similarity
  Dtype* video_sim = S_video_.mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                        batch_size_, batch_size_, feature_size_,
                        Dtype(1), video_centroid, video_centroid, Dtype(0), video_sim);

  // calculate video label indicator
  Dtype* indicator_video = I_video_.mutable_gpu_data();
  kernel_label_indicator<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_label, indicator_video, batch_size_);

  // calculate pairwise loss
  Dtype* pairwise_loss = pairwise_loss_.mutable_gpu_data();
  Dtype bias = this->blobs_[0]->cpu_data()[0];
  kernel_pairwise_loss<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(video_sim, indicator_video, pairwise_loss, 
                              pairwise_loss_.count(), margin_width_, bias);

  // sum pairwise loss
  Dtype loss;
  caffe_gpu_asum<Dtype>(pairwise_loss_.count(), pairwise_loss, &loss);
  if (normalize_) {
    loss /= batch_size_ * batch_size_;
  }
  top[0]->mutable_cpu_data()[0] = loss;

  // for (int i = 0; i < bottom[1]->count(); ++ i) {
  //   LOG(INFO) << "label: " << i << " " << bottom[1]->cpu_data()[i];
  // }

  // for (int i = 0; i < I_video_.count(); ++ i) {
  //   LOG(INFO) << "video label similarity: " << i << " " << I_video_.cpu_data()[i];
  // }

  // for (int i = 0; i < S_video_.count(); ++ i) {
  //   LOG(INFO) << "video similarity: " << i << " " << S_video_.cpu_data()[i];
  // }

  // LOG(INFO) << "bias " << this->blobs_[0]->cpu_data()[0];
}

template <typename Dtype>
__global__ void kernel_backprop(const Dtype* pairwise_loss, const Dtype* I_video, 
                                const Dtype* video_centroid, Dtype* out,
                                const int clip_size, const int batch_size,
                                const int feature_size) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < clip_size * batch_size) {
    int t = thread_id / batch_size , i = thread_id % batch_size;
    for (int j = 0; j < batch_size; ++ j) {
      Dtype weight = pairwise_loss[i * batch_size + j] > 0 ? 
        (-2 * I_video[i * batch_size + j] / clip_size) : 0;
      for (int d = 0; d < feature_size; ++ d) {
        out[(t * batch_size + i) * feature_size + d] += weight * video_centroid[j * feature_size + d];
      }
    }
  }
}

template <typename Dtype>
__global__ void kernel_bias_gradient(const Dtype* pairwise_loss, 
                                     const Dtype* label_indicator, 
                                     Dtype* out, const int batch_size) {
  Dtype gradient = 0;
  for (int i = 0; i < batch_size * batch_size; ++ i) {
    gradient += pairwise_loss[i] > 0 ? -label_indicator[i] : 0;
  }
  out[0] = gradient;
}

template <typename Dtype>
void MaxMarginAvgSimVideoLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
  const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* pairwise_loss = pairwise_loss_.gpu_data();
    const Dtype* I_video = I_video_.gpu_data();
    const Dtype* video_centroid = video_centroid_.gpu_data();

    // backprop residue to bottom
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(bottom[0]->count(), Dtype(0), bottom_diff);
    kernel_backprop<Dtype><<<CAFFE_GET_BLOCKS(clip_size_ * batch_size_),
      CAFFE_CUDA_NUM_THREADS>>>(pairwise_loss, I_video, video_centroid, bottom_diff, 
                                clip_size_, batch_size_, feature_size_);

    // backprop to bias term
    Dtype* bias_term = this->blobs_[0]->mutable_gpu_diff();
    kernel_bias_gradient<Dtype><<<1, 1>>>(pairwise_loss, I_video, bias_term, batch_size_);

    if (normalize_) {
      Dtype scale_factor = 1.0 / (batch_size_ * batch_size_);
      caffe_gpu_scal<Dtype>(bottom[0]->count(), scale_factor, bottom_diff);
      caffe_gpu_scal<Dtype>(1, scale_factor, bias_term);
    } 
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxMarginAvgSimVideoLossLayer);
} // namespace caffe
