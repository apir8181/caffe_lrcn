
#include "caffe/l_loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_video_label_indicator(const Dtype* label, Dtype* out, const int batch_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size * batch_size) {
    int i = id / batch_size;
    int j = id % batch_size;
    int label1 = (int) label[i];
    int label2 = (int) label[j];
    bool has_common = (label1 & label2) > 0;
    out[id] = has_common ? 1 : -1;
  }
}

template <typename Dtype>
__global__ void kernel_video_similarity(const Dtype *frame_sim, const Dtype* label_indicator, Dtype* out,
                                        const int clip_size, const int batch_size, const int instance_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size * batch_size) {
    int i = id / batch_size, j = id % batch_size;
    Dtype sim = 0;
    // for each timestamp find nearest neighbor
    for (int t = 0; t < clip_size; ++ t) {
      Dtype it_jth_max = -1e10, jt_ith_max = -1e10;
      for (int t_hat = 0; t_hat < clip_size; ++ t_hat) {
        int row_1 = t * batch_size + i, col_1 = t_hat * batch_size + j;
        int row_2 = t * batch_size + j, col_2 = t_hat * batch_size + i;
        int idx_1 = row_1 * instance_size + col_1, idx_2 = row_2 * instance_size + col_2;
        it_jth_max = fmax(it_jth_max, frame_sim[idx_1]);
        jt_ith_max = fmax(jt_ith_max, frame_sim[idx_2]);
      }
      sim += (it_jth_max + jt_ith_max) / clip_size;
    }
    out[id] = sim;
  }
}

template <typename Dtype>
__global__ void kernel_pairwise_loss(const Dtype* video_sim, const Dtype* label_indicator, Dtype* out,
                                     const int count, const Dtype margin_width, const Dtype bias) {
  CUDA_KERNEL_LOOP(i, count) {
    Dtype pre_loss = margin_width - label_indicator[i] * (video_sim[i] + bias);
    out[i] = pre_loss > 0 ? pre_loss : 0;
  }
}

template <typename Dtype>
void MaxMarginMaxSimVideoLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();

  // calculate per frame pair similarity
  Dtype* frame_sim = S_frame_.mutable_gpu_data();
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, 
                 instance_size_, instance_size_, feature_size_,
                 Dtype(1), bottom_data, bottom_data, Dtype(0), frame_sim);

  // calculate video label indicator
  Dtype* indicator_video = I_video_.mutable_gpu_data();
  kernel_video_label_indicator<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_label, indicator_video, batch_size_);

  // calculate video pair similarity
  Dtype* video_sim = S_video_.mutable_gpu_data();
  kernel_video_similarity<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(frame_sim, indicator_video, video_sim, 
                              clip_size_, batch_size_, instance_size_);

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

  LOG(INFO) << "bias " << this->blobs_[0]->cpu_data()[0];
}

template <typename Dtype>
__global__ void kernel_nearest(const Dtype* frame_sim, const Dtype* label_indicator,
                               const Dtype* pairwise_loss, const Dtype* bottom_data, 
                               Dtype* out,
                               const int clip_size, const int batch_size, const int instance_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < clip_size * batch_size) {
    int t = id / batch_size, i = id % batch_size;
    for (int j = 0; j < batch_size; ++ j) {
      // find nearest neighbor
      Dtype it_jth_max = 1e-10, it_jth_idx = 0;
      for (int t_hat = 0; t_hat < clip_size; ++ t_hat) {
        int row = t * batch_size + i, col = t_hat * batch_size + j;
        int idx = row * instance_size + col;
        if (it_jth_max < frame_sim[idx]) {
          it_jth_max = frame_sim[idx];
          it_jth_idx = t_hat;
        }
      }

      Dtype indicator = (pairwise_loss[i * batch_size + j] > 0);
      Dtype weight = indicator * label_indicator[i * batch_size + j] / clip_size;
      int row = t * batch_size + i, col = it_jth_idx * batch_size + j;
      out[row * instance_size + col] = -weight;
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
void MaxMarginMaxSimVideoLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
  const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* frame_sim = S_frame_.gpu_data();
    const Dtype* label_indicator = I_video_.gpu_data();
    const Dtype* pairwise_loss = pairwise_loss_.gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* nn_frame = N_frame_.mutable_gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    // get nearest neighbor
    caffe_gpu_set<Dtype>(N_frame_.count(), Dtype(0), nn_frame);    
    kernel_nearest<Dtype><<<CAFFE_GET_BLOCKS(clip_size_ * batch_size_),
      CAFFE_CUDA_NUM_THREADS>>>(frame_sim, label_indicator,
                                pairwise_loss, bottom_data, 
                                nn_frame,
                                clip_size_, batch_size_, instance_size_);
    // backprop to bottom
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          instance_size_, feature_size_, instance_size_,
                          Dtype(1), nn_frame, bottom_data, Dtype(0), bottom_diff);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                          instance_size_, feature_size_, instance_size_,
                          Dtype(1), nn_frame, bottom_data, Dtype(1), bottom_diff);

    // backprop to bias term
    Dtype* bias_term = this->blobs_[0]->mutable_gpu_diff();
    kernel_bias_gradient<Dtype><<<1, 1>>>(pairwise_loss, label_indicator, bias_term, batch_size_);

    if (normalize_) {
      Dtype scale_factor = 1.0 / (batch_size_ * batch_size_);
      caffe_gpu_scal<Dtype>(bottom[0]->count(), scale_factor, bottom_diff);
      caffe_gpu_scal<Dtype>(1, scale_factor, bias_term);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxMarginMaxSimVideoLossLayer);
} // namespace caffe
