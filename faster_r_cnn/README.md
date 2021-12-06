# Faster R-CNN with Pytorch

## Featrue map
- VGG-16 을 백본으로 사용
- VGG-16 : 
    * Layer 수 : 31
    * 출력층의 W, H 가 입력의 1/16인 layer 까지만 사용(30개 사용)
    * 30층 출력 : (512, 40, 40) - (C, W, H) # 입력 사이즈 640 * 640


## RPN(Region Proposal Network)

</br>

### Anchor
- 출력 feature map(512, 40, 40)의 grid cell 수 만큼의 anchor 중심점 생성 : 40 * 40개의 anchor 중심점
- 하나의 grid cell은 입력 이미지의 grid(16*16)에 해당하는 물체의 정보 포함
- 하나의 anchor 중심점은 ratio(3), scale(9)에 따라 여러개 존재 : 40 * 40 * 3 * 3 = 14400 개
- 총 40*40*3*3 개의 anchor는 feature map(512, 40, 40)에 대응되는 RPN의 출력 

### RPN layer
- VGG-16 feature map에 3 * 3 * 512 conv2d 적용 : intermediate layer 생성
- Intermediate layer에 1 * 1 conv2d 적용
    - 1) Classification : 1 * 1 * [2*(3*3)] conv2d 적용
    - 2) Bbox regression : 1 * 1 * [4*(3*3)] conv2d 적용
- RPN 출력
    - 1) Classification : 18 * 40 * 40 -> 14400 * 2 (not or exist)
    - 2) Bbox regression : 36 * 40 * 40 -> 14400 * 4 (x_c, y_c, w, h)

### Train RPN
- **학습에는 전체 후보 앵커(40*40*3*3) 중 256개를 사용**
- 256 개중 positive와 negative anchor 수는 1:1, 그러나 positive 수가 적을 경우 negative anchor수로 총 256개 맞춤
- Positive anchor
    - 1) GT box의 IOU가 가장 높은 anchor들
    - 2) GT box와 IOU가 0.7 이상인 anchor들
- Negative anchor
    - 1) GT box와의 IOU가 0.3 이하인 anchor들
- No contribution
    - 1) Postive와 negative가 아닌 anchor들은 학습에 사용 X
- Positive & Negative에서 128개가 넘는 앵커들 중 128개 랜덤 선택
- **Positive + negative anchor** = 256

### Loss RPN
- RPN 출력을 14400 * 2 또는 14400 * 4 로 바꾸어 loss 계산
- Classification loss : 
    - 1) Cross entropy 사용하여 anchor안에 객체가 있을 확률 학습
    - 2) RPN의 classification 출력 14400 * 2 중 **0 인덱스에는 없을 확률, 1 인덱스에는 있을 확률**
    - 3) 앞서 선정한 256개의 anchor만 학습에 사용
- Bbox regression loss :
    - 1) Smooth L1 loss 사용하여 박스 coordinates 학습
    - 2) 학습하는 coordinate는 GT box와의 차이를 학습하여, 현재 anchor에 물체가 포함될 수 있게 변환하는 값을 학습
    - 3) 앞서 선정한 256개의 anchor 중 positive anchor만 학습에 사용
    
### RPN to Proposal layer
- 학습된 RPN의 출력값을 사용하여 cls와 reg task를 위한 입력으로 만드는 방법
- RPN은 anchor 박스에 객체가 있을 확률(cls)와 anchor박스가 객체를 포함할 수 있게 변화할 값을 가지고 있음(reg)
- RPN 출력값을 사용하여 anchor(14400)들의 변환된 후보 anchor(roi)를 생성
- 이 때, 생성된 roi가 이미지 영역(640 * 640) 안에 포함되도록 변형 (np.clip 사용)
- 1개의 gird 영역을 설명하지 못하는(원래 이미지의 16 pixel) anchor 삭제
- 후보 anchor 중 객체가 있을 확률 순으로 **12000** 개의 anchor만 선택
- 선택된 후보 anchor 중 NMS를 적용

    

