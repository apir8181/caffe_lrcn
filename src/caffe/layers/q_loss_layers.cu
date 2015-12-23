
#include "caffe/q_loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_q_loss(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = log( cosh( fabs(data[index]) - 1 ) );
  }
}

template <typename Dtype>
void QLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int bottom_data_count = bottom[0]->count();
  CHECK_EQ(bottom_data_count, temp_.count()) << "Input data shape changed.";

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* temp = temp_.mutable_gpu_data();
  kernel_q_loss<Dtype><<<CAFFE_GET_BLOCKS(bottom_data_count),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data_count, bottom_data, temp);

  Dtype loss;
  caffe_gpu_asum<Dtype>(bottom_data_count, temp, &loss);
  if (normalize_) {
    int outer_num = bottom[0]->count(0, axis_);
    loss *= scale_ / outer_num;
  } else {
    loss *= scale_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void kernel_backprop(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    Dtype val = tanh( fabs(data[index]) - 1 );
    if (data[index] < 0) val *= -1;
    out[index] = val;
  }
}

template <typename Dtype>
void QLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
  const vector<Blob<Dtype>*>& bottom) {
  int bottom_data_count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  kernel_backprop<Dtype><<<CAFFE_GET_BLOCKS(bottom_data_count),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data_count, bottom_data, bottom_diff);
  if (normalize_) {
    Dtype scale_factor = scale_ / bottom[0]->count(0, axis_);
    caffe_gpu_scal<Dtype>(bottom[0]->count(), scale_factor, bottom_diff);
  } else {
    caffe_gpu_scal<Dtype>(bottom[0]->count(), scale_, bottom_diff);
  }
}

// Q Loss original
/*
template <typename Dtype>
__global__ void kernel_q_loss(const int count, const Dtype *data, Dtype* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = fabs( fabs(data[idx]) - 1 );
  }
}

template <typename Dtype>
__global__ void kernel_backprop(const int count, const Dtype* data, Dtype *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
     Dtype sign1 = fabs(data[idx]) - 1 > 0 ? 1 : -1;
     Dtype sign2 = fabs(data[idx]) > 0 ? 1 : -1;
     out[idx] = sign1 * sign2;
  }  
}
*/

INSTANTIATE_LAYER_GPU_FUNCS(QLossLayer);

} // namespace caffe
