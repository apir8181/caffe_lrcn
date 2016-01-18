
#include "caffe/l_loss_outer_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_frame_similarity(
  const Dtype* data, Dtype* out, 
  const int instance_size, const int feature_size) {
  
  CUDA_KERNEL_LOOP(ij, instance_size * instance_size) {
    Dtype val = 0;
    int i = ij / instance_size, j = ij % instance_size;
    for (int d = 0; d < feature_size; ++ d) {
      int id = i * feature_size + d, jd = j * feature_size + d;
      val += data[id] * data[jd];
    }
    val /= feature_size;
    out[ij] = val;
  }

}


template <typename Dtype>
__global__ void kernel_frame_indicator(
  const Dtype* label, Dtype* out, const int instance_size) {
  
  CUDA_KERNEL_LOOP(ij, instance_size * instance_size) {
    int i = ij / instance_size, j = ij % instance_size;
    int label1 = (int) label[i], label2 = (int) label[j];
    bool matched = (label1 & label2) > 0;
    out[ij] = matched ? 1 : -1;
  }

}


template <typename Dtype>
__global__ void kernel_pairwise_error(
    const Dtype* S_frame, const Dtype* I_frame,
    Dtype* out, const int count) {

   CUDA_KERNEL_LOOP(i, count) {
     Dtype pre_loss = I_frame[i] - S_frame[i];
     out[i] = pre_loss;
   }

}


template <typename Dtype>
__global__ void kernel_pairwise_loss(
    const Dtype* S_frame, const Dtype* I_frame,
    Dtype* out, const int count) {

   CUDA_KERNEL_LOOP(i, count) {
     Dtype pre_loss = I_frame[i] - S_frame[i];
     out[i] = pre_loss * pre_loss;
   }

}


template <typename Dtype>
void LLossOuterLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  Dtype* S_frame = S_frame_.mutable_gpu_data();
  Dtype* I_frame = I_frame_.mutable_gpu_data();
  Dtype* pairwise_error = pairwise_error_.mutable_gpu_data();
  Dtype* pairwise_loss = pairwise_loss_.mutable_gpu_data();

  int pair_size = instance_size_ * instance_size_;
  // calculate video pair similarity
  kernel_frame_similarity<Dtype><<<CAFFE_GET_BLOCKS(pair_size),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, S_frame, instance_size_, feature_size_);

  // calculate video label indicator
  kernel_frame_indicator<Dtype><<<CAFFE_GET_BLOCKS(pair_size),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_label, I_frame, instance_size_);

  // calculate pairwise error
  kernel_pairwise_error<Dtype><<<CAFFE_GET_BLOCKS(pair_size),
    CAFFE_CUDA_NUM_THREADS>>>(S_frame, I_frame, pairwise_error, pair_size);
                              
  kernel_pairwise_loss<Dtype><<<CAFFE_GET_BLOCKS(pair_size),
    CAFFE_CUDA_NUM_THREADS>>>(S_frame, I_frame, pairwise_loss, pair_size);

  Dtype loss, abs_error;
  caffe_gpu_asum<Dtype>(pair_size, pairwise_loss, &loss);
  caffe_gpu_asum<Dtype>(pair_size, pairwise_error, &abs_error);
  if (normalize_) {
    loss /= pair_size;
    abs_error /= pair_size;
  }
  LOG(INFO) << "loss:" << loss;
  LOG(INFO) << "abs_error:" << abs_error;

  top[0]->mutable_cpu_data()[0] = loss;

  // for (int i = 0; i < batch_size_ * batch_size_; ++ i) {
  //   LOG(INFO) << " ij:" << i
  //             << " loss:" << pairwise_loss_.mutable_cpu_data()[i]
  //             << " label:" << I_video_.mutable_cpu_data()[i]
  //             << " sim:" << S_video_.mutable_cpu_data()[i];
  // }
}

template <typename Dtype>
__global__ void kernel_backprop(
  const Dtype* pairwise_error, const Dtype* data, Dtype* out,
  const int instance_size, int feature_size) {

  CUDA_KERNEL_LOOP(id, instance_size * feature_size) {
    int i = id / feature_size, d = id % feature_size;
    Dtype val = 0;
    for (int j = 0; j < instance_size; ++ j) {
      int ij = i * instance_size + j;
      int jd = j * feature_size + d;
      Dtype weight = -2 * pairwise_error[ij];
      if (i != j) {
        val += weight * data[jd];
      } else {
        val += weight * data[id] * 2;
      }
    }
    out[id] = val / feature_size;
  }
}


template <typename Dtype>
void LLossOuterLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
  const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int bottom_count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* pairwise_error = pairwise_error_.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    kernel_backprop<Dtype><<<CAFFE_GET_BLOCKS(instance_size_ * feature_size_),
      CAFFE_CUDA_NUM_THREADS>>>(pairwise_error, bottom_data, bottom_diff,
                                instance_size_, feature_size_);

    if (normalize_) {
      Dtype scale_factor = 1.0 / (instance_size_ * instance_size_);
      caffe_gpu_scal<Dtype>(bottom[0]->count(), scale_factor, bottom_diff);
    } 

    Dtype bottom_diff_som, bottom_data_som;
    caffe_gpu_asum<Dtype>(bottom_count, bottom_data, &bottom_data_som);
    caffe_gpu_asum<Dtype>(bottom_count, bottom_diff, &bottom_diff_som);
    bottom_data_som /= bottom_count;
    bottom_diff_som /= bottom_count;
    LOG(INFO) << "bottom data SOM:" << bottom_data_som;
    LOG(INFO) << "bottom diff SOM:" << bottom_diff_som;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LLossOuterLayer);
} // namespace caffe
