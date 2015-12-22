#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// code by qiaoan
namespace caffe {

template <typename Dtype>
VideoDataLayer<Dtype>::VideoDataLayer(const LayerParameter& param)
    : BaseDataLayer<Dtype>(param), prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(top.size() == 3) << "output size should be 3(data, lable, marker).";
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  const int clip_size = this->layer_param_.video_data_param().clip_size();
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  string root_folder = this->layer_param_.video_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames, clip range, and labels
  const string& source = this->layer_param_.video_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.video_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    ShuffleData();
  }
  LOG(INFO) << "A total of " << lines_.size() << " videos.";

  lines_id_ = 0;
  // Read an video, and use it to initialize the top blob.
  vector<cv::Mat> clip = ReadVideoFrames(
    root_folder + lines_[lines_id_].first, clip_size, new_height, new_width);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(clip[0]);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the clip_size and batch_size.
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  CHECK_GT(clip_size, 0) << "Positive clip size required";
  top_shape[0] = clip_size * batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  vector<int> label_shape(1, clip_size * batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  LOG(INFO) << "output data label size: " << top[1]->num();

  // marker
  vector<int> marker_shape(1, clip_size * batch_size);
  top[2]->Reshape(marker_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].marker_.Reshape(marker_shape);
  }  
  LOG(INFO) << "output data marker size: " << top[2]->num();

  // Allocate space and start Internal Thread
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleData() {
  const unsigned int seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(seed));
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      TripleBatch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void VideoDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  TripleBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // copy loaded labels.
    top[1]->ReshapeLike(batch->label_);
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
    // copy to loaded markers.
    top[2]->ReshapeLike(batch->marker_);
    caffe_copy(batch->marker_.count(), batch->marker_.cpu_data(),
               top[2]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

// This function is called on prefetch thread
template <typename Dtype>
void VideoDataLayer<Dtype>::load_batch(TripleBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  const int clip_size = this->layer_param_.video_data_param().clip_size();
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  string root_folder = this->layer_param_.video_data_param().root_folder();
  // Reshape according to the first video of each batch
  // on single input batches allows for inputs of varying dimension.
  vector<cv::Mat> clip = ReadVideoFrames(
    root_folder + lines_[lines_id_].first, clip_size, new_height, new_width);
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(clip[0]);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the clip_size and batch_size.
  top_shape[0] = clip_size * batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_marker = batch->marker_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    vector<cv::Mat> clip = ReadVideoFrames(
      root_folder + lines_[lines_id_].first, clip_size, new_height, new_width);
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations to the each frame in video
    for (int frame_id = 0; frame_id < clip_size; ++frame_id) {
      int item_id = frame_id * batch_size + batch_id;
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(clip[frame_id], &(this->transformed_data_));
      trans_time += timer.MicroSeconds();
      prefetch_label[item_id] = lines_[lines_id_].second;
      prefetch_marker[item_id] = frame_id == 0 ? 0 : 1;
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleData();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(VideoDataLayer, Forward);
#endif

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
