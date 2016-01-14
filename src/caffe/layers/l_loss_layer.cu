
#include "caffe/l_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#ifndef LLOSS_ZERO
#define LLOSS_ZERO 0
#endif

namespace caffe {

template <typename Dtype>
__global__ void kernel_frame_length(
  const Dtype* data, Dtype* out, 
  const int instance_size, const int feature_size) {

  CUDA_KERNEL_LOOP(ti, instance_size) {
    Dtype val = 0;
    for (int d = 0; d < feature_size; ++ d) {
      int idx = ti * feature_size + d;
      val += data[idx] * data[idx];
    }
    out[ti] = sqrt(val);
  }

}


template <typename Dtype>
__global__ void kernel_label_indicator(
  const Dtype* label, Dtype* out, const int batch_size) {

  CUDA_KERNEL_LOOP(ij, batch_size * batch_size) {
    int i = ij / batch_size, j = ij % batch_size;
    int label1 = (int) label[i];
    int label2 = (int) label[j];
    bool has_common = (label1 & label2) > 0;
    out[ij] = has_common ? 1 : -1;
  }

}


 template <typename Dtype>
 __global__ void kernel_frame_similarity(
   const Dtype* data, const Dtype* L_frame, Dtype* out, 
   const int instance_size, const int feature_size) {

   CUDA_KERNEL_LOOP(ij, instance_size * instance_size) {
     int i = ij / instance_size, j = ij % instance_size;
     Dtype val = 0;
     Dtype weight = L_frame[i] * L_frame[j];
     for (int d = 0; d < feature_size; ++ d) {
       //val += data[i*feature_size + d] * data[j*feature_size + d] / weight;
       val += data[i*feature_size + d] * data[j*feature_size + d];
     }
     out[ij] = val;
   }
   
}

template <typename Dtype>
__global__ void kernel_video_similarity(
  const Dtype* S_frame, Dtype* out,
  const int clip_size, const int batch_size, const int instance_size) {

  CUDA_KERNEL_LOOP(ij, batch_size * batch_size) {
    int i = ij / batch_size, j = ij % batch_size;
    Dtype val = 0;
    for (int t = 0; t < clip_size; ++ t) {
      for (int th = 0; th < clip_size; ++ th) {
        int ti = t*batch_size+i, thj = th*batch_size+j;
        val += S_frame[ti*instance_size+thj] / clip_size / clip_size;
      }
    }
    out[ij] = val;
  }

}


template <typename Dtype>
__global__ void kernel_pairwise_loss(
  const Dtype* S_video, const Dtype* I_video, const Dtype* bias, Dtype* out,
  const int batch_size, const Dtype margin_width) {

  CUDA_KERNEL_LOOP(ij, batch_size * batch_size) {
    int i = ij / batch_size, j = ij % batch_size;
    //Dtype pre_loss = margin_width - I_video[ij] * (S_video[ij] + bias[0]);
    Dtype pre_loss = margin_width - I_video[ij] * S_video[ij];
    out[ij] = pre_loss > LLOSS_ZERO ? pre_loss: 0;
  }

}

template <typename Dtype>
__global__ void kernel_accuracy(
  const Dtype* pairwise_loss, Dtype* out, const int count) {

  CUDA_KERNEL_LOOP(i, count) {
    out[i] = pairwise_loss[i] > LLOSS_ZERO ? 0 : 1;
  }

}


template <typename Dtype>
void LLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();

  Dtype* L_frame = L_frame_.mutable_gpu_data();
  kernel_frame_length<Dtype><<<CAFFE_GET_BLOCKS(instance_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, L_frame, instance_size_, feature_size_);

  // calculate per frame pair similarity
  Dtype* S_frame = S_frame_.mutable_gpu_data();
  kernel_frame_similarity<Dtype><<<CAFFE_GET_BLOCKS(instance_size_ * instance_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, L_frame, S_frame, 
                              instance_size_, feature_size_);

  // calculate video label indicator
  Dtype* I_video = I_video_.mutable_gpu_data();
  kernel_label_indicator<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_label, I_video, batch_size_);

  // calculate video pair similarity
  Dtype* S_video = S_video_.mutable_gpu_data();
  kernel_video_similarity<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(S_frame, S_video,
                              clip_size_, batch_size_, instance_size_);

  // calculate pairwise loss
  //Dtype* bias = this->blobs_[0]->mutable_gpu_data();
  Dtype* pairwise_loss = pairwise_loss_.mutable_gpu_data();
  kernel_pairwise_loss<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(S_video, I_video, 0, pairwise_loss, 
                              batch_size_, margin_width_);

  Dtype* accuracy = accuracy_.mutable_gpu_data();
  kernel_accuracy<Dtype><<<CAFFE_GET_BLOCKS(batch_size_ * batch_size_),
    CAFFE_CUDA_NUM_THREADS>>>(pairwise_loss, accuracy, batch_size_ * batch_size_);

  // sum pairwise loss
  Dtype loss;
  caffe_gpu_asum<Dtype>(pairwise_loss_.count(), pairwise_loss, &loss);
  top[0]->mutable_cpu_data()[0] = loss / (batch_size_ * batch_size_);

  // classifier accuracy
  // Dtype ac;
  // caffe_gpu_asum<Dtype>(accuracy_.count(), accuracy, &ac);
  // ac /= batch_size_ * batch_size_;
  // LOG(INFO) << "accuracy:" << ac;

  // for (int i = 0; i < S_frame_.count(); ++ i)
  //    LOG(INFO) << "frame similarity: " << i << " " << S_frame_.cpu_data()[i];
  // for (int i = 0; i < L_frame_.count(); ++ i)
  //    LOG(INFO) << "frame length: " << i << " " << L_frame_.cpu_data()[i];
  // for (int i = 0; i < I_video_.count(); ++ i)
  //   LOG(INFO) << "video label : " << i << " " << I_video_.cpu_data()[i];
  // for (int i = 0; i < S_video_.count(); ++ i)
  //    LOG(INFO) << "video sim : " << i << " " << S_video_.cpu_data()[i];
  for (int i = 0; i < pairwise_loss_.count(); ++ i)
    LOG(INFO) << i << " video_sim:" << S_video_.cpu_data()[i]
              << " label:" << I_video_.cpu_data()[i]
              << " point score:" << S_video_.cpu_data()[i] * I_video_.cpu_data()[i]
              << " loss:" << pairwise_loss_.cpu_data()[i];
  //LOG(INFO) << "bias " << this->blobs_[0]->mutable_cpu_data()[0];
}


template <typename Dtype>
__global__ void kernel_backprop(
  const Dtype* data, const Dtype* L_frame, const Dtype* S_frame,
  const Dtype* I_video, const Dtype* pairwise_loss, Dtype* out,
  const int clip_size, const int batch_size,
  const int instance_size, const int feature_size) {

  CUDA_KERNEL_LOOP(tid, instance_size * feature_size) {
    int ti = tid / feature_size;
    int i = ti % batch_size;
    int d = tid % feature_size;

    Dtype val = 0;
    for (int th = 0; th < clip_size; ++ th)
      for (int j = 0; j < batch_size; ++ j) {
        int ij = i*batch_size + j;
        int thj = th*batch_size + j;
        int thjd = thj*feature_size + d;
        int tithj = ti*instance_size + thj;
        //if (pairwise_loss[ij] <= LLOSS_ZERO) continue;
        //Dtype weight = -2.0 * I_video[ij] / clip_size / clip_size;
        //Dtype l_part = data[thjd] / L_frame[ti] / L_frame[thj];
        //Dtype r_part = data[tid] * S_frame[tithj] / L_frame[ti] / L_frame[ti];
        //val += weight * (l_part - r_part);
        Dtype weight = -2.0 * I_video[ij] / clip_size / clip_size;
        val += weight * data[thjd];
      }

    out[tid] = val;
  }

} 

template <typename Dtype>
__global__ void kernel_bias_backprop(
  const Dtype* pairwise_loss, const Dtype* I_label,
  Dtype* bias_diff, const int count) {
  
  CUDA_KERNEL_LOOP(j, 1) {
    Dtype val = 0;
    for (int i = 0; i < count; ++ i) {
      val += pairwise_loss[i] > 0 ? -I_label[i] : 0;
    }
    bias_diff[0] = val;
  }

}


template <typename Dtype>
void LLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
  const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* L_frame = L_frame_.gpu_data();
    const Dtype* S_frame = S_frame_.gpu_data();
    const Dtype* I_video = I_video_.gpu_data();
    const Dtype* pairwise_loss = pairwise_loss_.gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const int bottom_count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    // backprop to bottom
    kernel_backprop<Dtype><<<CAFFE_GET_BLOCKS(instance_size_ * feature_size_),
      CAFFE_CUDA_NUM_THREADS>>>(bottom_data, L_frame, S_frame,
                                I_video, pairwise_loss, bottom_diff,
                                clip_size_, batch_size_, 
                                instance_size_, feature_size_);

    // // backprop to bias
    // Dtype* bias_diff = this->blobs_[0]->mutable_gpu_diff();
    // kernel_bias_backprop<Dtype><<<CAFFE_GET_BLOCKS(1),
    //     CAFFE_CUDA_NUM_THREADS>>>(pairwise_loss, I_video, 
    //                               bias_diff, batch_size_ * batch_size_);

    if (normalize_) {
         Dtype scale_factor = 1.0 / (batch_size_ * batch_size_);
         caffe_gpu_scal<Dtype>(bottom[0]->count(), scale_factor, bottom_diff);
    //     caffe_gpu_scal<Dtype>(1, scale_factor, bias_diff);
    }

    Dtype bottom_diff_som, bottom_data_som;
    caffe_gpu_asum<Dtype>(bottom_count, bottom_data, &bottom_data_som);
    caffe_gpu_asum<Dtype>(bottom_count, bottom_diff, &bottom_diff_som);
    bottom_data_som /= bottom_count;
    bottom_diff_som /= bottom_count;
    LOG(INFO) << "bottom data SOM:" << bottom_data_som;
    LOG(INFO) << "bottom diff SOM:" << bottom_diff_som;
    //LOG(INFO) << "bias diff:" << this->blobs_[0]->mutable_cpu_diff()[0];
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LLossLayer);
} // namespace caffe
