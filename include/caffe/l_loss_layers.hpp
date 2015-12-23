#ifndef CAFFE_L_LOSS_LAYERS_HPP_
#define CAFFE_L_LOSS_LAYERS_HPP_

#include "caffe/blob.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
class MaxMarginMaxSimVideoLossLayer : public LossLayer<Dtype> {
public:
  explicit MaxMarginMaxSimVideoLossLayer(const LayerParameter& param) 
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
        
  virtual inline const char* type() const { return "MaxMarginMaxSimVideoLoss"; } 
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
        
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);

private:
  int clip_size_, batch_size_, instance_size_, feature_size_;
  bool normalize_;
  Dtype margin_width_;
  Blob<Dtype> S_frame_; // per frame similarity for @f$ (ti, \hat{t} j) @f$
  Blob<Dtype> I_frame_; // per frame label sim indicators for @f$ (ti, \hat{t} j) @f$
  Blob<Dtype> N_frame_; // pair frame nearest neighbors indicator for @f$ (ti, \hat{t} j) @f$
  Blob<Dtype> S_video_; // per video similarity for @f$ (i, j) @f$
  Blob<Dtype> I_video_; // per video label sim indicators
  Blob<Dtype> pairwise_loss_;
  Blob<Dtype> temp_frame_;
};

template <typename Dtype>
class MaxMarginSimpleSimVideoLossLayer: public LossLayer<Dtype> {
public:
  explicit  MaxMarginSimpleSimVideoLossLayer(const LayerParameter& param) 
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
        
  virtual inline const char* type() const { return "MaxMarginSimpleSimVideoLoss"; } 
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
        
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, 
                            const vector<Blob<Dtype>*>& bottom);

private:
  int clip_size_, batch_size_, instance_size_, feature_size_;
  bool normalize_;
  Dtype margin_width_;
  Blob<Dtype> S_video_;
  Blob<Dtype> I_video_;
  Blob<Dtype> pairwise_loss_;
};

  
} // namespace caffe

#endif // CAFFE_L_LOSS_LAYERS_HPP_
