
#include "caffe/l_loss_layers.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void FramePairLabelIndicator(
    const int clip_size, const int batch_size, const int instance_size,
    const Dtype *label, Dtype* I) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < instance_size * instance_size) {
      int row = idx / instance_size;
      int col = idx % instance_size;
      int i = row % batch_size;
      int j = col % batch_size;
      if (i == j) return;

      int label1 = (int) label[row];
      int label2 = (int) label[col];
      I[idx] = label1 == label2 ? 1 : -1;
  }
}

template <typename Dtype>
__global__ void FramePairNearestIndicator(
    const int clip_size, const int batch_size, const int instance_size,
    const Dtype* S_frame, Dtype *N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < instance_size * batch_size) {
    int t = id / batch_size / batch_size;
    int i = id / batch_size % batch_size;
    int j = id % batch_size;
    if (i == j) return;

    // find max t_hat
    int row = t * batch_size + i;
    Dtype max_val = Dtype(-1e10);
    Dtype t_hat = 0;
    for (int l = 0; l < clip_size; ++ l) {
      int col = l * batch_size + j;
      Dtype val = S_frame[row * instance_size + col];
      if (val > max_val) {
        max_val = val;
        t_hat = l;
      }
    }
    int col = t_hat * batch_size + j;
    N[row * instance_size + col] = 1;
  }
}

template <typename Dtype>
__global__ void VideoPairLabelIndicator(
    const int clip_size, const int batch_size, const int instance_size,
    const Dtype* label, Dtype* S_video) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size * batch_size) {
    int i = id / batch_size;
    int j = id % batch_size;
    if (i == j) return;
    S_video[i * batch_size + j] = label[i] == label[j] ? 1 : -1;
  }
}

template <typename Dtype>
__global__ void VideoPairSimilarity(
    const int clip_size, const int batch_size, const int instance_size, const int feature_size,
    const Dtype* N_frame, const Dtype* S_frame, Dtype* S_video) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < batch_size * batch_size) {
    int i = id / batch_size;
    int j = id % batch_size;
    int video_idx = i * batch_size + j;
    if (i == j) return;

    for (int t = 0; t < clip_size; ++ t) {
      for (int t_hat = 0; t_hat < clip_size; ++ t_hat) {
        int row = t * batch_size + i;
        int col = t_hat * batch_size + j;
        int idx = row * instance_size + col;
        int inv_idx = col * instance_size + row;
        S_video[video_idx] += N_frame[idx] * S_frame[idx] / clip_size;
        S_video[video_idx] += N_frame[inv_idx] * S_frame[inv_idx] / clip_size;
      }
    }
  }
}

template <typename Dtype>
__global__ void PairwiseLoss(const int count, const Dtype margin_width,
                             const Dtype* S_video, const Dtype* I_video, Dtype* out) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < count) {
    Dtype pre_loss = I_video[id] * (margin_width - S_video[id]);
    out[id] = pre_loss > 0 ? pre_loss : 0;
  }
}    

template <typename Dtype>
void MaxMarginMaxSimVideoLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();

  // calculate per frame pair similarity
  Dtype* frame_sim = S_frame_.mutable_gpu_data();
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, 
                 instance_size_, instance_size_, feature_size_,
                 Dtype(1), data, data, Dtype(0), frame_sim);

  // calculate frame label indicator
  Dtype* indicator_frame = I_frame_.mutable_gpu_data();
  caffe_gpu_set(I_frame_.count(), Dtype(0), indicator_frame);
  FramePairLabelIndicator<Dtype><<<CAFFE_GET_BLOCKS(instance_size_ * instance_size_), 
    CAFFE_CUDA_NUM_THREADS>>>(clip_size_, batch_size_, instance_size_, label, indicator_frame);

  // calculate nearest frame
  Dtype* nn_indicator = N_frame_.mutable_gpu_data();
  caffe_gpu_set(N_frame_.count(), Dtype(0), nn_indicator);
  FramePairNearestIndicator<Dtype><<<CAFFE_GET_BLOCKS(instance_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(clip_size_, batch_size_, instance_size_, frame_sim, nn_indicator);
  
  // calculate video pair similarity
  Dtype* video_sim = S_video_.mutable_gpu_data();
  caffe_gpu_set(S_video_.count(), Dtype(0), video_sim);
  VideoPairSimilarity<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(clip_size_, batch_size_, instance_size_, feature_size_, nn_indicator, frame_sim, video_sim);

  // calculate video label indicator
  Dtype* indicator_video = I_video_.mutable_gpu_data();
  caffe_gpu_set(I_video_.count(), Dtype(0), indicator_video);
  VideoPairLabelIndicator<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(clip_size_, batch_size_, instance_size_, label, indicator_video);

  // calculate pairwise loss
  Dtype* pairwise_loss = pairwise_loss_.mutable_gpu_data();
  caffe_gpu_set<Dtype>(pairwise_loss_.count(), Dtype(0), pairwise_loss);
  PairwiseLoss<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(pairwise_loss_.count(), margin_width_, video_sim, indicator_video, pairwise_loss);

  // sum pairwise loss
  Dtype loss;
  caffe_gpu_asum<Dtype>(pairwise_loss_.count(), pairwise_loss, &loss);
  if (normalize_) {
    loss /= batch_size_ * (batch_size_ - 1);
  }
  top[0]->mutable_cpu_data()[0] = loss;

  /*
  for (int i = 0; i < I_video_.count(); ++ i) {
    LOG(INFO) << "video label similarity "
              << i << " " << I_video_.cpu_data()[i];
  }  

  for (int i = 0; i < S_video_.count(); ++ i) {
    LOG(INFO) << "video similarity "
              << i << " " << S_video_.cpu_data()[i];
  }  

  for (int i = 0; i < pairwise_loss_.count(); ++ i) {
    LOG(INFO) << "Pairwise loss "
              << i << " " << pairwise_loss_.cpu_data()[i];
  }
  */
}

template <typename Dtype>
__global__ void FramePairBPScaleMatrix(
  const int clip_size, const int batch_size, const int instance_size, const int feature_size,
  const Dtype* nn_indicator, const Dtype* frame_indicator, const Dtype* pairwise_loss, 
  Dtype* out) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < instance_size * instance_size) {
    int row = id / instance_size, col = id % instance_size;
    int i = row % batch_size, j = col % batch_size;
    out[id] = pairwise_loss[i * batch_size + j] > 0 ? 
      -frame_indicator[id] * nn_indicator[id] / clip_size : 0;
  }
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
    Dtype* temp = temp_frame_.mutable_gpu_data();
    FramePairBPScaleMatrix<Dtype><<<CAFFE_GET_BLOCKS(instance_size_ * instance_size_),
      CAFFE_CUDA_NUM_THREADS>>>(clip_size_, batch_size_, instance_size_, feature_size_,
                                N_frame_.gpu_data(), I_frame_.gpu_data(), 
                                pairwise_loss_.gpu_data(), temp);

    const Dtype* data = bottom[0]->gpu_data();
    Dtype* diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          instance_size_, feature_size_, instance_size_,
                          Dtype(1), temp, data, Dtype(0), diff);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                          instance_size_, feature_size_, instance_size_,
                          Dtype(1), temp, data, Dtype(1), diff);
    if (normalize_) {
      Dtype scale_factor = batch_size_ * (batch_size_ - 1);
      caffe_gpu_scal<Dtype>(bottom[0]->count(), 1.0 / scale_factor, diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxMarginMaxSimVideoLossLayer);
} // namespace caffe
