# Manual Download Guide for Brainstem Datasets

*Last Updated: 2025-09-11*

## ✅ Successfully Downloaded

### Allen Brain Atlas
- **Status**: ✅ Complete
- **Location**: `/data/datasets/brainstem_segmentation/raw/allen/metadata/`
- **Content**: 21 experiments with brainstem-relevant developmental stages (E11.5, E13.5, E15.5, E18.5)
- **Sample Images**: 2 section images downloaded to `/raw/allen/`
- **Next Steps**: Use the metadata to fetch specific image sections as needed

## ⚠️ Manual Download Required

### DevCCF (Developmental Common Coordinate Framework) - PRIORITY 1

The DevCCF atlases are critical for registration templates. Please download manually:

#### Primary Sources

1. **Nature Communications Article (2024)**:
   - URL: https://www.nature.com/articles/s41467-024-45070-z
   - Look for supplementary data links
   - Contains the latest DevCCF reference atlases
   
2. **BrainImageLibrary Repository**:
   - Main site: https://www.brainimagelibrary.org/
   - Search for: "DevCCF" or "Kronman"
   - Project: Developmental Common Coordinate Framework
   - Authors: Kronman et al., 2024
   
3. **Allen Brain Map Community Forum**:
   - URL: https://community.brain-map.org/t/developmental-common-coordinate-framework/571
   - Contains download links and documentation
   - Look for posts by Allen Institute staff

#### Files to Download

Download these specific files for each developmental stage:

**E11.5 Atlas**:
- `E11.5_DevCCF_Annotations.nii.gz` - Anatomical labels
- `E11.5_DevCCF_Reference.nii.gz` - Grayscale reference volume
- Size: ~50-100 MB each

**E13.5 Atlas**:
- `E13.5_DevCCF_Annotations.nii.gz` - Anatomical labels
- `E13.5_DevCCF_Reference.nii.gz` - Grayscale reference volume
- Size: ~50-100 MB each

**E15.5 Atlas**:
- `E15.5_DevCCF_Annotations.nii.gz` - Anatomical labels
- `E15.5_DevCCF_Reference.nii.gz` - Grayscale reference volume
- Size: ~50-100 MB each

#### Where to Place Files

Store all downloaded DevCCF files in:
```
/Users/camdouglas/quark/data/datasets/brainstem_segmentation/templates/devccf/
```

### Alternative Sources

If the primary sources are unavailable, try:

1. **Zenodo Repository**:
   - Search: https://zenodo.org
   - Keywords: "DevCCF mouse brain" or "Kronman developmental atlas"
   
2. **FigShare**:
   - Search: https://figshare.com
   - Keywords: "Developmental Common Coordinate Framework"
   
3. **GitHub Releases**:
   - Check: https://github.com/AllenInstitute
   - Look for DevCCF-related repositories
   
4. **Direct Contact**:
   - Email corresponding authors from Kronman et al., 2024
   - Request access through Allen Institute contact form

### Additional Optional Datasets

#### GUDMAP Database
For supplementary MRI volumes:

1. Visit: https://www.gudmap.org/
2. Create free account (required for downloads)
3. Search: "mouse brain" AND ("E11.5" OR "E13.5" OR "E15.5")
4. Filter by: 
   - Species: Mouse
   - Data Type: 3D imaging
   - Format: NIFTI
5. Download selected volumes
6. Store in: `/data/datasets/brainstem_segmentation/raw/gudmap/`

#### EBRAINS Atlas Resources
For high-resolution histology:

1. Visit: https://ebrains.eu/
2. Navigate to: Data & Knowledge → Atlases
3. Search: "developmental mouse brain"
4. Filter: Embryonic stages
5. Download histology sections
6. Store in: `/data/datasets/brainstem_segmentation/raw/ebrains/`

#### BrainMaps.org
For ultra-high resolution sections:

1. Visit: http://brainmaps.org/
2. Browse: Mouse → Developmental
3. Select: E12-E18 stages
4. Use their API or web interface for downloads
5. Store in: `/data/datasets/brainstem_segmentation/raw/brainmaps/`

## Sample Allen Brain Atlas Image URLs

The following URLs can be used to download additional sample images:
```bash
# Download sample sections (already have 2, these are additional)
curl -O "http://api.brain-map.org/api/v2/section_image_download/101383769"
curl -O "http://api.brain-map.org/api/v2/section_image_download/101383770"
curl -O "http://api.brain-map.org/api/v2/section_image_download/101383771"
```

## Verification Checklist

After downloading, verify you have:

**Essential (Required)**:
- [ ] DevCCF E11.5 annotations (.nii.gz)
- [ ] DevCCF E13.5 annotations (.nii.gz)
- [ ] DevCCF E15.5 annotations (.nii.gz)
- [ ] DevCCF E11.5 reference (.nii.gz)
- [ ] DevCCF E13.5 reference (.nii.gz)
- [ ] DevCCF E15.5 reference (.nii.gz)

**Already Complete**:
- [x] Allen metadata (21 experiments)
- [x] Sample Allen images (2 sections)
- [x] Registration configuration
- [x] Nucleus catalog (26 nuclei)

**Optional Enhancements**:
- [ ] GUDMAP MRI volumes
- [ ] EBRAINS histology sections
- [ ] BrainMaps high-res images
- [ ] Additional Allen sections

## Validation

After downloading, run the validation script to check data integrity:

```bash
cd /Users/camdouglas/quark
python brain/modules/brainstem_segmentation/validate_data.py
```

Expected output should show:
- ✅ Allen: 21 experiments
- ✅ DevCCF: 3/3 stages found
- ✅ Registration: ANTs configured

## Next Steps

Once downloads are complete:

1. **Validate data integrity**: Run the validation script above
2. **Test registration pipeline**: Align a sample volume to DevCCF template
3. **Proceed to Step 3**: Draft JSON label schema using nucleus catalog
4. **Begin Step 4**: Risk/feasibility analysis

## Troubleshooting

### If DevCCF Downloads Fail

1. **Check Allen Community Forum**: Often has mirror links
2. **Use Web Archive**: Check archive.org for cached versions
3. **Contact Authors**: Email from the Nature paper
4. **Use Alternative Atlas**: Allen CCFv3 adult atlas as temporary substitute

### File Format Issues

- If files are in `.nrrd` format, convert using:
  ```bash
  # Requires SimpleITK or ANTs
  python -c "import SimpleITK as sitk; img = sitk.ReadImage('file.nrrd'); sitk.WriteImage(img, 'file.nii.gz')"
  ```

- If files are compressed differently (`.tar.gz`, `.zip`):
  ```bash
  tar -xzf archive.tar.gz  # for .tar.gz
  unzip archive.zip        # for .zip
  ```

## Support

For questions about the datasets or download process:
- Check the task documentation at `/state/tasks/roadmap_tasks/brainstem_segmentation_tasks.md`
- Review the nucleus catalog at `/data/datasets/brainstem_segmentation/metadata/`
- Consult the roadmap at `/management/rules/roadmap/stage1_embryonic_rules.md`

---

*This guide is part of the Brainstem Segmentation project (Stage 1 Embryonic Roadmap)*
