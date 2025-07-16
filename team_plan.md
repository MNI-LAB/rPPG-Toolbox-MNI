# ML Team Plan (July 9 – August 20, 2025)

## Overall Goals

- [ ] Test iBVP-Net with MR-NIRP Dataset
- [ ] Evaluate single channel performance (targeting 3-4 BPM MAE)
- [ ] Try incorporating depth into the model
- [ ] Compare depth compensation methods with 2D images

---

## Weekly Plan

### Week 1 (July 9 – July 15)

**Focus**: iBVP-Net preprocessing and verification
**Tasks**:

- [x] Finish dataset preparation on greatlakes
- [x] Align oximeter data with camera data
- [x] Start training process
- [x] Tune the loss function (Negative Pearson Loss) ([based on rppg-toolbox repo issues](https://github.com/ubicomplab/rPPG-Toolbox/issues/254))
- [ ] Decent train/loss curves

---

### Week 2 (July 16 – July 22)

**Focus**: iBVP-Net with our own data
**Tasks**:

- [ ] Ensure iBVP-Net is applicable to IR data
- [ ] Record and feed our own data into iBVP-Net
- [ ] Verify the model's performance on our dataset
- [ ] See if pretrained model suffice or training with our data is necessary
- [ ] Compare with our conventional CVSM pipeline

---

### Week 3-4 (July 23 – August 5)

**Focus**: Depth Incorporation
**Tasks**:

- [ ] Explore depth compensation methods
- [ ] Investigate the impact of depth on model performance

**Possible Depth Compensation Methods**:

- [ ] compensate depth after iBVP-Net gives rppg output
- [ ] use depth as an additional input to iBVP-Net
- [ ] develop individual branch for depth in iBVP-Net
- [ ] Dump time segments with large depth variations

---

### Week 5 (August 6 – August 12)

**Focus**: Experiments with depth compensation methods
**Tasks**:

- [ ] Record data with depth variations
- [ ] Implement and test different depth compensation methods
- [ ] Evaluate the performance of each method
- [ ] Analyze results and compare with baseline performance

---

### Week 6 (August 13 – August 20)

**Focus**: Demo and Finalization
**Tasks**:

- [ ] Prepare demo for the team
- [ ] a GUI to feed in data and visualize results

---
