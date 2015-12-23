
#include "caffe/l_loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void VideoSim(const Dtype* data, Dtype* out, 
                         const int clip_size, const int batch_size, const int feature_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size * batch_size) {
    int i = id / batch_size, j = id % batch_size;
    Dtype sim = 0;
    for (int t = 0; t < clip_size; ++ t) {
      for (int d = 0; d < feature_size; ++ d) {
        int idx1 = (t * batch_size + i) * feature_size + d;
        int idx2 = (t * batch_size + j) * feature_size + d;
        sim += data[idx1] * data[idx2] / clip_size;
      }
    }
    out[id] = sim;      
  }
}

template <typename Dtype>
__global__ void VideoLabelIndicator(const Dtype* label, Dtype* out, const int batch_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size * batch_size) {
    int i = id / batch_size, j = id % batch_size;
    out[id] = label[i] == label[j] ? 1 : -1;
  }
}

template <typename Dtype>
__global__ void PairwiseLoss(const Dtype* S_video, const Dtype* I_video, Dtype* out, 
                             const int count, const Dtype margin_width) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < count) {
    Dtype pre_loss = margin_width - I_video[id] * S_video[id];
    out[id] = pre_loss > 0 ? pre_loss : 0;
  }
}    

template <typename Dtype>
void MaxMarginSimpleSimVideoLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  
  // calculate video pair similarity
  Dtype* video_sim = S_video_.mutable_gpu_data();
  VideoSim<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(data, video_sim, clip_size_, batch_size_, feature_size_);

  // calculate video label indicator
  Dtype* indicator_video = I_video_.mutable_gpu_data();
  VideoLabelIndicator<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(label, indicator_video, batch_size_);

  // calculate pairwise loss
  Dtype* pairwise_loss = pairwise_loss_.mutable_gpu_data();
  PairwiseLoss<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(video_sim, indicator_video, pairwise_loss, 
                              pairwise_loss_.count(), margin_width_);

  // sum pairwise loss
  Dtype loss;
  caffe_gpu_asum<Dtype>(pairwise_loss_.count(), pairwise_loss, &loss);
  if (normalize_) {
    loss /= batch_size_ * batch_size_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void Backward(const Dtype* pairwise_loss, const Dtype* I_video, 
                         const Dtype* data, Dtype* out, 
                         const int clip_size, const int batch_size, 
                         const int feature_size, const Dtype margin_width) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < clip_size * batch_size) {
    int t = id / batch_size, i = id % batch_size;
    for (int j = 0; j < batch_size; ++ j) {
      Dtype weight = pairwise_loss[i * batch_size + j] > 0 ? I_video[i * batch_size + j] : 0;
      for (int d = 0; d < feature_size; ++ d) {
        int idx1 = (t * batch_size + i) * feature_size + d;
        int idx2 = (t * batch_size + j) * feature_size + d;
        out[idx1] -= 2.0 / clip_size * weight * data[idx2];
      }
    }
  }
}

template <typename Dtype>
void MaxMarginSimpleSimVideoLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
  const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* pairwise_loss = pairwise_loss_.mutable_gpu_data();
    const Dtype* I_video = I_video_.mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    caffe_gpu_set<Dtype>(bottom[0]->count(), Dtype(0), bottom_diff);
    Backward<Dtype><<<CAFFE_GET_BLOCKS(clip_size_ * batch_size_),
      CAFFE_CUDA_NUM_THREADS>>>(pairwise_loss, I_video, 
                                bottom_data, bottom_diff, 
                                clip_size_, batch_size_, 
                                feature_size_, margin_width_);

    if (normalize_) {
      Dtype scale_factor = batch_size_ * batch_size_;
      caffe_gpu_scal<Dtype>(bottom[0]->count(), 1.0 / scale_factor, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxMarginSimpleSimVideoLossLayer);
} // namespace caffe
