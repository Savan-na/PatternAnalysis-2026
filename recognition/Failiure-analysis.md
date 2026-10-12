### Case Study 1 — B040_Week0: Improved Bladder Segmentation with Persistent Prostate Omission

A paired qualitative evaluation was performed on validation case B040_Week0 using the Standard and Improved 3D U-Net models after ten training epochs. Both models were evaluated on the same resampled MRI volume using 64³ sliding windows with a stride of 32.

Three slices (Z = 51, 62, and 73) were selected across the ground-truth prostate extent to examine segmentation behaviour at different spatial positions.

The Improved U-Net produced more spatially continuous Body predictions and substantially improved Bladder segmentation. Bladder Dice increased from 0.5365 to 0.8695. This improvement was primarily associated with an increase in Bladder recall from 36.9% to 80.7%, while precision decreased slightly from 98.6% to 94.2%.

However, neither model successfully segmented the Prostate, Rectum, or Bones in this case. The Improved model assigned no voxels to these three classes, despite their presence in the ground-truth segmentation.

For the Prostate specifically, all 7,105 ground-truth voxels were assigned to other anatomical classes, demonstrating complete class omission rather than a minor boundary-localisation error.

This case illustrates an important limitation of the proposed architectural improvements: better segmentation of larger anatomical regions does not necessarily translate into successful recognition of smaller or more challenging structures.

The observations support a diagnosis of partial class prediction collapse, but do not independently establish whether its cause is class imbalance, patch sampling, loss design, or insufficient training.

These findings describe validation behaviour only and should not be interpreted as final held-out test performance.