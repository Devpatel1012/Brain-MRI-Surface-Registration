

#---------------------------------
# New invocation of recon-all Mon Mar 16 14:12:57 IST 2026 

 mri_convert /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/OASIS_OAS1_0001_MR1_nii/orig.nii.gz /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig/001.mgz 

#--------------------------------------------
#@# MotionCor Mon Mar 16 14:13:03 IST 2026

 cp /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig/001.mgz /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/rawavg.mgz 


 mri_info /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/rawavg.mgz 


 mri_convert /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/rawavg.mgz /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig.mgz --conform 


 mri_add_xform_to_header -c /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/transforms/talairach.xfm /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig.mgz /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig.mgz 


 mri_info /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig.mgz 


 mri_synthstrip --threads 1 -i /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig.mgz -o /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/synthstrip.mgz 


 mri_synthseg --i /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig.mgz --o /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/synthseg.rca.mgz --threads 1 --vol /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/stats/synthseg.vol.csv --keepgeom --addctab --cpu 


 fs-synthmorph-reg --s OASIS_0001 --threads 1 --i /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/orig.mgz --test 

#--------------------------------------------
#@# Nu Intensity Correction Mon Mar 16 14:26:07 IST 2026

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --n 2 --ants-n4 


 mri_add_xform_to_header -c /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Mon Mar 16 14:27:47 IST 2026

 mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz 

#--------------------------------------

#@# MCADura Segmentation Mon Mar 16 14:28:45 IST 2026
#--------------------------------------

#@# VSinus Segmentation Mon Mar 16 14:29:01 IST 2026

 mri_mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/T1.mgz /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/synthstrip.mgz /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/brainmask.mgz 

#-------------------------------------
#@# EM Registration Mon Mar 16 14:29:21 IST 2026

 mri_em_register -uns 3 -mask brainmask.mgz nu.mgz /usr/local/freesurfer/8.1.0/average/RB_all_2020-01-02.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize Mon Mar 16 14:34:34 IST 2026

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /usr/local/freesurfer/8.1.0/average/RB_all_2020-01-02.gca transforms/talairach.lta norm.mgz 

#--------------------------------------

#@# EntoWM Segmentation Mon Mar 16 14:35:11 IST 2026
#--------------------------------------
#@# CC Seg Mon Mar 16 14:35:22 IST 2026

 seg2cc --s OASIS_0001 

#--------------------------------------
#@# Merge ASeg Mon Mar 16 14:35:39 IST 2026

 cp aseg.auto.mgz aseg.presurf.mgz 

#--------------------------------------------
#@# Intensity Normalization2 Mon Mar 16 14:35:39 IST 2026

 mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Mon Mar 16 14:37:03 IST 2026

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 


 mri_mask -oval 1 -invert brain.finalsurfs.mgz /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/mca-dura.mgz brain.finalsurfs.mgz 


 mri_mask -oval 1 -invert brain.finalsurfs.mgz /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/vsinus.mgz brain.finalsurfs.mgz 


 mri_edit_wm_with_aseg -sa-fix-ento-wm entowm.mgz 2 255 255 brain.finalsurfs.mgz brain.finalsurfs.mgz 


 mri_edit_wm_with_aseg -sa-fix-acj aseg.presurf.mgz 255 255 brain.finalsurfs.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Mon Mar 16 14:37:09 IST 2026

 AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz 


 mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in -fix-ento-wm entowm.mgz 3 255 255 -fix-acj aseg.presurf.mgz 255 255 -fill-seg-wm -fix-scm-ha 1 wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz 

Fixing entowm in wm.mgz

 mri_edit_wm_with_aseg -sa-fix-ento-wm entowm.mgz 3 255 255 wm.mgz wm.mgz 

Fixing ACJ in wm.mgz

 mri_edit_wm_with_aseg -sa-fix-acj aseg.presurf.mgz 255 255 wm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Mon Mar 16 14:38:35 IST 2026

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.presurf.mgz -ctab /usr/local/freesurfer/8.1.0/SubCorticalMassLUT.txt wm.mgz filled.mgz 

 cp filled.mgz filled.auto.mgz
#--------------------------------------------
#@# Tessellate lh Mon Mar 16 14:39:09 IST 2026

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Tessellate rh Mon Mar 16 14:39:14 IST 2026

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Mon Mar 16 14:39:19 IST 2026

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Smooth1 rh Mon Mar 16 14:39:22 IST 2026

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Mon Mar 16 14:39:25 IST 2026

 mris_inflate -no-save-sulc ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# Inflation1 rh Mon Mar 16 14:39:35 IST 2026

 mris_inflate -no-save-sulc ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Mon Mar 16 14:39:45 IST 2026

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# QSphere rh Mon Mar 16 14:40:42 IST 2026

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#@# Fix Topology lh Mon Mar 16 14:41:35 IST 2026

 mris_fix_topology -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 -threads 1 OASIS_0001 lh 

#@# Fix Topology rh Mon Mar 16 14:43:12 IST 2026

 mris_fix_topology -threads 1 -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 -threads 1 OASIS_0001 rh 


 mris_euler_number ../surf/lh.orig.premesh 


 mris_euler_number ../surf/rh.orig.premesh 


 mris_remesh --remesh --iters 3 --input /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.orig.premesh --output /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.orig 


 mris_remesh --remesh --iters 3 --input /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.orig.premesh --output /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.orig 


 mris_remove_intersection ../surf/lh.orig ../surf/lh.orig 


 rm -f ../surf/lh.inflated 


 mris_remove_intersection ../surf/rh.orig ../surf/rh.orig 


 rm -f ../surf/rh.inflated 

#--------------------------------------------
#@# AutoDetGWStats lh Mon Mar 16 14:46:47 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_autodet_gwstats --o ../surf/autodet.gw.stats.lh.dat --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/lh.orig.premesh
#--------------------------------------------
#@# AutoDetGWStats rh Mon Mar 16 14:46:50 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_autodet_gwstats --o ../surf/autodet.gw.stats.rh.dat --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/rh.orig.premesh
#--------------------------------------------
#@# WhitePreAparc lh Mon Mar 16 14:46:53 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --wm wm.mgz --threads 1 --invol brain.finalsurfs.mgz --lh --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white --seg aseg.presurf.mgz --restore-255 --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 --outvol mrisps.wpa.mgz
#--------------------------------------------
#@# WhitePreAparc rh Mon Mar 16 14:49:10 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --wm wm.mgz --threads 1 --invol brain.finalsurfs.mgz --rh --i ../surf/rh.orig --o ../surf/rh.white.preaparc --white --seg aseg.presurf.mgz --restore-255 --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 --outvol mrisps.wpa.mgz
#--------------------------------------------
#@# CortexLabel lh Mon Mar 16 14:51:25 IST 2026
#--------------------------------------------
#@# CortexLabel+HipAmyg lh Mon Mar 16 14:51:41 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mri_label2label --label-cortex ../surf/lh.white.preaparc aseg.presurf.mgz 1 ../label/lh.cortex+hipamyg.label
#--------------------------------------------
#@# CortexLabel rh Mon Mar 16 14:51:50 IST 2026
#--------------------------------------------
#@# CortexLabel+HipAmyg rh Mon Mar 16 14:52:07 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mri_label2label --label-cortex ../surf/rh.white.preaparc aseg.presurf.mgz 1 ../label/rh.cortex+hipamyg.label
#--------------------------------------------
#@# Smooth2 lh Mon Mar 16 14:52:15 IST 2026

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white.preaparc ../surf/lh.smoothwm 

#--------------------------------------------
#@# Smooth2 rh Mon Mar 16 14:52:19 IST 2026

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white.preaparc ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh Mon Mar 16 14:52:22 IST 2026

 mris_inflate ../surf/lh.smoothwm ../surf/lh.inflated 

#--------------------------------------------
#@# Inflation2 rh Mon Mar 16 14:52:34 IST 2026

 mris_inflate ../surf/rh.smoothwm ../surf/rh.inflated 

#--------------------------------------------
#@# Curv .H and .K lh Mon Mar 16 14:52:45 IST 2026

 mris_curvature -w -seed 1234 lh.white.preaparc 


 mris_curvature -seed 1234 -thresh .999 -n -a 5 -w -distances 10 10 lh.inflated 

#--------------------------------------------
#@# Curv .H and .K rh Mon Mar 16 14:53:15 IST 2026

 mris_curvature -w -seed 1234 rh.white.preaparc 


 mris_curvature -seed 1234 -thresh .999 -n -a 5 -w -distances 10 10 rh.inflated 

#--------------------------------------------
#@# Sphere lh Mon Mar 16 14:53:46 IST 2026

 mris_sphere -threads 1 -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Sphere rh Mon Mar 16 14:57:57 IST 2026

 mris_sphere -threads 1 -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 

#--------------------------------------------
#@# Surf Reg lh Mon Mar 16 15:01:57 IST 2026

 mris_register -curv -threads 1 ../surf/lh.sphere /usr/local/freesurfer/8.1.0/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/lh.sphere.reg 


 ln -sf lh.sphere.reg lh.fsaverage.sphere.reg 

#--------------------------------------------
#@# Surf Reg rh Mon Mar 16 15:05:04 IST 2026

 mris_register -curv -threads 1 ../surf/rh.sphere /usr/local/freesurfer/8.1.0/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/rh.sphere.reg 


 ln -sf rh.sphere.reg rh.fsaverage.sphere.reg 

#--------------------------------------------
#@# Jacobian white lh Mon Mar 16 15:07:57 IST 2026

 mris_jacobian ../surf/lh.white.preaparc ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# Jacobian white rh Mon Mar 16 15:07:59 IST 2026

 mris_jacobian ../surf/rh.white.preaparc ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

#--------------------------------------------
#@# AvgCurv lh Mon Mar 16 15:08:01 IST 2026

 mrisp_paint -a 5 /usr/local/freesurfer/8.1.0/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/lh.sphere.reg ../surf/lh.avg_curv 

#--------------------------------------------
#@# AvgCurv rh Mon Mar 16 15:08:03 IST 2026

 mrisp_paint -a 5 /usr/local/freesurfer/8.1.0/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/rh.sphere.reg ../surf/rh.avg_curv 

#-----------------------------------------
#@# Cortical Parc lh Mon Mar 16 15:08:05 IST 2026

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 OASIS_0001 lh ../surf/lh.sphere.reg /usr/local/freesurfer/8.1.0/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.annot 

#-----------------------------------------
#@# Cortical Parc rh Mon Mar 16 15:08:12 IST 2026

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 OASIS_0001 rh ../surf/rh.sphere.reg /usr/local/freesurfer/8.1.0/average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.annot 

#--------------------------------------------
#@# WhiteSurfs lh Mon Mar 16 15:08:20 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 1 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white.preaparc --o ../surf/lh.white --white --nsmooth 0 --rip-label ../label/lh.cortex.label --rip-bg --rip-surf ../surf/lh.white.preaparc --aparc ../label/lh.aparc.annot --restore-255 --restore-255 --outvol mrisps.white.mgz --rip-bg-lof
#--------------------------------------------
#@# WhiteSurfs rh Mon Mar 16 15:10:18 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --threads 1 --wm wm.mgz --invol brain.finalsurfs.mgz --rh --i ../surf/rh.white.preaparc --o ../surf/rh.white --white --nsmooth 0 --rip-label ../label/rh.cortex.label --rip-bg --rip-surf ../surf/rh.white.preaparc --aparc ../label/rh.aparc.annot --restore-255 --restore-255 --outvol mrisps.white.mgz --rip-bg-lof
#--------------------------------------------
#@# T1PialSurf lh Mon Mar 16 15:12:11 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 1 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white --o ../surf/lh.pial.T1 --pial --nsmooth 0 --rip-label ../label/lh.cortex+hipamyg.label --pin-medial-wall ../label/lh.cortex.label --aparc ../label/lh.aparc.annot --repulse-surf ../surf/lh.white --white-surf ../surf/lh.white --restore-255
#--------------------------------------------
#@# T1PialSurf rh Mon Mar 16 15:14:21 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --threads 1 --wm wm.mgz --invol brain.finalsurfs.mgz --rh --i ../surf/rh.white --o ../surf/rh.pial.T1 --pial --nsmooth 0 --rip-label ../label/rh.cortex+hipamyg.label --pin-medial-wall ../label/rh.cortex.label --aparc ../label/rh.aparc.annot --repulse-surf ../surf/rh.white --white-surf ../surf/rh.white --restore-255
#@# white curv lh Mon Mar 16 15:16:25 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --curv-map ../surf/lh.white 2 10 ../surf/lh.curv
#@# white area lh Mon Mar 16 15:16:27 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --area-map ../surf/lh.white ../surf/lh.area
#@# pial curv lh Mon Mar 16 15:16:28 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --curv-map ../surf/lh.pial 2 10 ../surf/lh.curv.pial
#@# pial area lh Mon Mar 16 15:16:30 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --area-map ../surf/lh.pial ../surf/lh.area.pial
#@# thickness lh Mon Mar 16 15:16:32 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --thickness ../surf/lh.white ../surf/lh.pial 20 5 ../surf/lh.thickness
#@# area and vertex vol lh Mon Mar 16 15:16:49 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
#@# white curv rh Mon Mar 16 15:16:53 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --curv-map ../surf/rh.white 2 10 ../surf/rh.curv
#@# white area rh Mon Mar 16 15:16:55 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --area-map ../surf/rh.white ../surf/rh.area
#@# pial curv rh Mon Mar 16 15:16:57 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --curv-map ../surf/rh.pial 2 10 ../surf/rh.curv.pial
#@# pial area rh Mon Mar 16 15:17:00 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --area-map ../surf/rh.pial ../surf/rh.area.pial
#@# thickness rh Mon Mar 16 15:17:02 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri
mris_place_surface --thickness ../surf/rh.white ../surf/rh.pial 20 5 ../surf/rh.thickness
#@# area and vertex vol rh Mon Mar 16 15:17:18 IST 2026
cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri

#-----------------------------------------
#@# Curvature Stats lh Mon Mar 16 15:17:23 IST 2026

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/lh.curv.stats -F smoothwm OASIS_0001 lh curv sulc 


#-----------------------------------------
#@# Curvature Stats rh Mon Mar 16 15:17:27 IST 2026

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/rh.curv.stats -F smoothwm OASIS_0001 rh curv sulc 

#--------------------------------------------
#@# Cortical ribbon mask Mon Mar 16 15:17:30 IST 2026

 mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon --parallel OASIS_0001 

#-----------------------------------------
#@# Cortical Parc 2 lh Mon Mar 16 15:18:19 IST 2026

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 OASIS_0001 lh ../surf/lh.sphere.reg /usr/local/freesurfer/8.1.0/average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 2 rh Mon Mar 16 15:18:29 IST 2026

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 OASIS_0001 rh ../surf/rh.sphere.reg /usr/local/freesurfer/8.1.0/average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 3 lh Mon Mar 16 15:18:39 IST 2026

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 OASIS_0001 lh ../surf/lh.sphere.reg /usr/local/freesurfer/8.1.0/average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Cortical Parc 3 rh Mon Mar 16 15:18:46 IST 2026

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 OASIS_0001 rh ../surf/rh.sphere.reg /usr/local/freesurfer/8.1.0/average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# WM/GM Contrast lh Mon Mar 16 15:18:54 IST 2026

 pctsurfcon --s OASIS_0001 --lh-only 

#-----------------------------------------
#@# WM/GM Contrast rh Mon Mar 16 15:19:03 IST 2026

 pctsurfcon --s OASIS_0001 --rh-only 

#-----------------------------------------
#@# Relabel Hypointensities Mon Mar 16 15:19:11 IST 2026

 mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz 

#-----------------------------------------
#@# APas-to-ASeg Mon Mar 16 15:19:21 IST 2026

 mri_surf2volseg --o aseg.mgz --i aseg.presurf.hypos.mgz --fix-presurf-with-ribbon /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/mri/ribbon.mgz --threads 1 --lh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.cortex.label --lh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.white --lh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.pial --rh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.cortex.label --rh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.white --rh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.pial 


 mri_brainvol_stats --subject OASIS_0001 

#-----------------------------------------
#@# AParc-to-ASeg aparc Mon Mar 16 15:19:38 IST 2026

 mri_surf2volseg --o aparc+aseg.mgz --label-cortex --i aseg.mgz --threads 1 --lh-annot /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.aparc.annot 1000 --lh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.cortex.label --lh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.white --lh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.pial --rh-annot /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.aparc.annot 2000 --rh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.cortex.label --rh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.white --rh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.pial 

#-----------------------------------------
#@# AParc-to-ASeg aparc.a2009s Mon Mar 16 15:20:24 IST 2026

 mri_surf2volseg --o aparc.a2009s+aseg.mgz --label-cortex --i aseg.mgz --threads 1 --lh-annot /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.aparc.a2009s.annot 11100 --lh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.cortex.label --lh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.white --lh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.pial --rh-annot /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.aparc.a2009s.annot 12100 --rh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.cortex.label --rh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.white --rh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.pial 

#-----------------------------------------
#@# AParc-to-ASeg aparc.DKTatlas Mon Mar 16 15:21:09 IST 2026

 mri_surf2volseg --o aparc.DKTatlas+aseg.mgz --label-cortex --i aseg.mgz --threads 1 --lh-annot /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.aparc.DKTatlas.annot 1000 --lh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.cortex.label --lh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.white --lh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.pial --rh-annot /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.aparc.DKTatlas.annot 2000 --rh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.cortex.label --rh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.white --rh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.pial 

#-----------------------------------------
#@# WMParc Mon Mar 16 15:21:55 IST 2026

 mri_surf2volseg --o wmparc.mgz --label-wm --i aparc+aseg.mgz --threads 1 --lh-annot /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.aparc.annot 3000 --lh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/lh.cortex.label --lh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.white --lh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/lh.pial --rh-annot /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.aparc.annot 4000 --rh-cortex-mask /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/label/rh.cortex.label --rh-white /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.white --rh-pial /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/surf/rh.pial 


 mri_segstats --seed 1234 --seg mri/wmparc.mgz --sum stats/wmparc.stats --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject OASIS_0001 --surf-wm-vol --ctab /usr/local/freesurfer/8.1.0/WMParcStatsLUT.txt --etiv --stiv /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/stats/synthseg.tiv.dat 

#-----------------------------------------
#@# Parcellation Stats lh Mon Mar 16 15:23:57 IST 2026

 mris_anatomical_stats -no-th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab OASIS_0001 lh white 


 mris_anatomical_stats -no-th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.pial.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab OASIS_0001 lh pial 

#-----------------------------------------
#@# Parcellation Stats rh Mon Mar 16 15:24:06 IST 2026

 mris_anatomical_stats -no-th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab OASIS_0001 rh white 


 mris_anatomical_stats -no-th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.pial.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab OASIS_0001 rh pial 

#-----------------------------------------
#@# Parcellation Stats 2 lh Mon Mar 16 15:24:15 IST 2026

 mris_anatomical_stats -no-th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.a2009s.stats -b -a ../label/lh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab OASIS_0001 lh white 

#-----------------------------------------
#@# Parcellation Stats 2 rh Mon Mar 16 15:24:20 IST 2026

 mris_anatomical_stats -no-th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.a2009s.stats -b -a ../label/rh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab OASIS_0001 rh white 

#-----------------------------------------
#@# Parcellation Stats 3 lh Mon Mar 16 15:24:25 IST 2026

 mris_anatomical_stats -no-th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.DKTatlas.stats -b -a ../label/lh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab OASIS_0001 lh white 

#-----------------------------------------
#@# Parcellation Stats 3 rh Mon Mar 16 15:24:29 IST 2026

 mris_anatomical_stats -no-th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.DKTatlas.stats -b -a ../label/rh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab OASIS_0001 rh white 

#--------------------------------------------
#@# ASeg Stats Mon Mar 16 15:24:34 IST 2026

 mri_segstats --seed 1234 --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --stiv /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/OASIS_0001/stats/synthseg.tiv.dat --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /usr/local/freesurfer/8.1.0/ASegStatsLUT.txt --subject OASIS_0001 

INFO: fsaverage subject does not exist in SUBJECTS_DIR
INFO: Creating symlink to fsaverage subject...

 cd /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles; ln -s /usr/local/freesurfer/8.1.0/subjects/fsaverage; cd - 

#--------------------------------------------
#@# BA_exvivo Labels lh Mon Mar 16 15:25:32 IST 2026

 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA1_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA2_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA3a_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA3a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA3b_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA3b_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA4a_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA4a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA4p_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA4p_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA6_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA6_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA44_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA44_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA45_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.BA45_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.V1_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.V1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.V2_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.V2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.MT_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.MT_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.entorhinal_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.entorhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.perirhinal_exvivo.label --trgsubject OASIS_0001 --trglabel ./lh.perirhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.FG1.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./lh.FG1.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.FG2.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./lh.FG2.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.FG3.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./lh.FG3.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.FG4.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./lh.FG4.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.hOc1.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./lh.hOc1.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.hOc2.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./lh.hOc2.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.hOc3v.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./lh.hOc3v.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.hOc4v.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./lh.hOc4v.mpm.vpnl.label --hemi lh --regmethod surface 


 mris_label2annot --s OASIS_0001 --ctab /usr/local/freesurfer/8.1.0/average/colortable_vpnl.txt --hemi lh --a mpm.vpnl --maxstatwinner --noverbose --l lh.FG1.mpm.vpnl.label --l lh.FG2.mpm.vpnl.label --l lh.FG3.mpm.vpnl.label --l lh.FG4.mpm.vpnl.label --l lh.hOc1.mpm.vpnl.label --l lh.hOc2.mpm.vpnl.label --l lh.hOc3v.mpm.vpnl.label --l lh.hOc4v.mpm.vpnl.label 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA1_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA2_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA3a_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA3a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA3b_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA3b_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA4a_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA4a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA4p_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA4p_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA6_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA6_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA44_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA44_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.BA45_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.BA45_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.V1_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.V1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.V2_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.V2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.MT_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.MT_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.entorhinal_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.entorhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/lh.perirhinal_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./lh.perirhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mris_label2annot --s OASIS_0001 --hemi lh --ctab /usr/local/freesurfer/8.1.0/average/colortable_BA.txt --l lh.BA1_exvivo.label --l lh.BA2_exvivo.label --l lh.BA3a_exvivo.label --l lh.BA3b_exvivo.label --l lh.BA4a_exvivo.label --l lh.BA4p_exvivo.label --l lh.BA6_exvivo.label --l lh.BA44_exvivo.label --l lh.BA45_exvivo.label --l lh.V1_exvivo.label --l lh.V2_exvivo.label --l lh.MT_exvivo.label --l lh.perirhinal_exvivo.label --l lh.entorhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s OASIS_0001 --hemi lh --ctab /usr/local/freesurfer/8.1.0/average/colortable_BA_thresh.txt --l lh.BA1_exvivo.thresh.label --l lh.BA2_exvivo.thresh.label --l lh.BA3a_exvivo.thresh.label --l lh.BA3b_exvivo.thresh.label --l lh.BA4a_exvivo.thresh.label --l lh.BA4p_exvivo.thresh.label --l lh.BA6_exvivo.thresh.label --l lh.BA44_exvivo.thresh.label --l lh.BA45_exvivo.thresh.label --l lh.V1_exvivo.thresh.label --l lh.V2_exvivo.thresh.label --l lh.MT_exvivo.thresh.label --l lh.perirhinal_exvivo.thresh.label --l lh.entorhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -no-th3 -mgz -f ../stats/lh.BA_exvivo.stats -b -a ./lh.BA_exvivo.annot -c ./BA_exvivo.ctab OASIS_0001 lh white 


 mris_anatomical_stats -no-th3 -mgz -f ../stats/lh.BA_exvivo.thresh.stats -b -a ./lh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab OASIS_0001 lh white 

#--------------------------------------------
#@# BA_exvivo Labels rh Mon Mar 16 15:28:15 IST 2026

 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA1_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA2_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA3a_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA3a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA3b_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA3b_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA4a_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA4a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA4p_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA4p_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA6_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA6_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA44_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA44_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA45_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.BA45_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.V1_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.V1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.V2_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.V2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.MT_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.MT_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.entorhinal_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.entorhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.perirhinal_exvivo.label --trgsubject OASIS_0001 --trglabel ./rh.perirhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.FG1.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./rh.FG1.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.FG2.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./rh.FG2.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.FG3.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./rh.FG3.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.FG4.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./rh.FG4.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.hOc1.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./rh.hOc1.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.hOc2.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./rh.hOc2.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.hOc3v.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./rh.hOc3v.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.hOc4v.mpm.vpnl.label --trgsubject OASIS_0001 --trglabel ./rh.hOc4v.mpm.vpnl.label --hemi rh --regmethod surface 


 mris_label2annot --s OASIS_0001 --ctab /usr/local/freesurfer/8.1.0/average/colortable_vpnl.txt --hemi rh --a mpm.vpnl --maxstatwinner --noverbose --l rh.FG1.mpm.vpnl.label --l rh.FG2.mpm.vpnl.label --l rh.FG3.mpm.vpnl.label --l rh.FG4.mpm.vpnl.label --l rh.hOc1.mpm.vpnl.label --l rh.hOc2.mpm.vpnl.label --l rh.hOc3v.mpm.vpnl.label --l rh.hOc4v.mpm.vpnl.label 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA1_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA2_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA3a_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA3a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA3b_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA3b_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA4a_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA4a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA4p_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA4p_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA6_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA6_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA44_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA44_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.BA45_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.BA45_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.V1_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.V1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.V2_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.V2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.MT_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.MT_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.entorhinal_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.entorhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /mnt/c/Users/ADMIN/Documents/Dev_Projects/OASIS_OAS1_0001_MR1_nii-20260316T061137Z-1-001/PreprocessFiles/fsaverage/label/rh.perirhinal_exvivo.thresh.label --trgsubject OASIS_0001 --trglabel ./rh.perirhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mris_label2annot --s OASIS_0001 --hemi rh --ctab /usr/local/freesurfer/8.1.0/average/colortable_BA.txt --l rh.BA1_exvivo.label --l rh.BA2_exvivo.label --l rh.BA3a_exvivo.label --l rh.BA3b_exvivo.label --l rh.BA4a_exvivo.label --l rh.BA4p_exvivo.label --l rh.BA6_exvivo.label --l rh.BA44_exvivo.label --l rh.BA45_exvivo.label --l rh.V1_exvivo.label --l rh.V2_exvivo.label --l rh.MT_exvivo.label --l rh.perirhinal_exvivo.label --l rh.entorhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s OASIS_0001 --hemi rh --ctab /usr/local/freesurfer/8.1.0/average/colortable_BA_thresh.txt --l rh.BA1_exvivo.thresh.label --l rh.BA2_exvivo.thresh.label --l rh.BA3a_exvivo.thresh.label --l rh.BA3b_exvivo.thresh.label --l rh.BA4a_exvivo.thresh.label --l rh.BA4p_exvivo.thresh.label --l rh.BA6_exvivo.thresh.label --l rh.BA44_exvivo.thresh.label --l rh.BA45_exvivo.thresh.label --l rh.V1_exvivo.thresh.label --l rh.V2_exvivo.thresh.label --l rh.MT_exvivo.thresh.label --l rh.perirhinal_exvivo.thresh.label --l rh.entorhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -no-th3 -mgz -f ../stats/rh.BA_exvivo.stats -b -a ./rh.BA_exvivo.annot -c ./BA_exvivo.ctab OASIS_0001 rh white 


 mris_anatomical_stats -no-th3 -mgz -f ../stats/rh.BA_exvivo.thresh.stats -b -a ./rh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab OASIS_0001 rh white 

