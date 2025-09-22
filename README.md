## Motivation

Age-related Macular Degeneration (AMD) and Macular Hole are leading causes of vision loss.  
Timely detection of retinal biomarkers such as drusen, subretinal/intraretinal fluid, and tissue defects is critical for diagnosis and treatment. Manual annotation of OCT scans is time-consuming and subjective, motivating the need for automated deep learning–based segmentation.

<div>
  <table>
    <tr>
      <td align="center">
        <img src="figures/motivation/amd.png" alt="Normal vs AMD retina" width="400"/><br>
        <b>Normal vs AMD retina</b>
      </td>
      <td align="center">
        <img src="figures/motivation/Macular_hole.png" alt="Normal vs Macular Hole retina" width="400"/><br>
        <b>Normal vs Macular Hole retina</b>
      </td>
    </tr>
  </table>
</div>

## Dataset & Preprocessing

**Datasets used**  
This project evaluates two OCT datasets: **AMD** and **Macular Hole**. Each dataset contains manually annotated B-scans with pixel-wise masks for clinically relevant biomarkers (drusen, intra-/sub-retinal fluid, tissue defects). A small set of sample images is included in `data/sample_images/` for quick inspection.

<div>
  <table>
    <tr>
      <td align="center">
        <img src="figures/dataset/dataset_dist.png" alt="Dataset overview" width="420"/><br>
        <b>Dataset overview</b>
      </td>
      <td align="center">
        <img src="figures/dataset/merged_annotation.png" alt="Annotation example" width="420"/><br>
        <b>Annotation (input / mask)</b>
      </td>
    </tr>
  </table>
</div>

**Preprocessing & augmentation**  
Images were standardized with intensity normalization and resized/cropped to fixed input dimensions. Typical pipeline steps: cropping to region-of-interest, intensity scaling, and data augmentation (random flips, rotations, contrast variations) during training to improve generalization.

<div>
  <table>
    <tr>
      <td align="center">
        <img src="figures/preprocessing/preprocessing_steps.png" alt="Preprocessing" width="420"/><br>
        <b>Preprocessing pipeline</b>
      </td>
      <td align="center">
        <img src="figures/preprocessing/augmentation_examples.png" alt="Augmentation examples" width="420"/><br>
        <b>Augmentation examples</b>
      </td>
    </tr>
  </table>
</div>
