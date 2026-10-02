/* =====================================================================
   ███  EDIT THIS FILE TO SWAP IN YOUR OWN ASSETS  ███
   ---------------------------------------------------------------------
   This is the ONLY file you need to touch to replace the placeholders
   with your real images and videos. Nothing else needs to change.

   HOW IT WORKS
   ------------
   • Every entry can use an IMAGE, a VIDEO, or stay a PLACEHOLDER.
   • Leave `src` empty ("")  -> a labelled placeholder box is shown.
   • Set `src` to an image    (.png/.jpg/.webp) -> shows the image.
   • Set `src` to a video      (.mp4/.webm)      -> shows a looping,
                                                    hover-to-play video.
   • `poster` (optional) is a still frame shown before a video plays.

   FOLDER LAYOUT
   -------------
   Clips live in assets/webvids/, one folder per results section, and each
   file is named after the slot it fills: <what>_<material>_<pred|gt>.mp4.

   assets/webvids/
     1_latent_prediction/   qualitative_<rigid|fluid|smoke>_<pred|gt>.mp4
                            multiview_input_multi.mp4
                            multiview_input_single_<m45|m27|p27|p45>.mp4
     2_rgb_decoder/         decoded_<rigid|smoke|occlusion>_<pred|gt>.mp4
                            multiview_rigid_<m45|m27|p27|p45>_<pred|gt>.mp4
                            multiview_fluid_cam<000|001|011|012>_pred.mp4
                            multiview_smoke_<m20|m10|p10|p20>_pred.mp4
     3_flow/                flow2d_<rigid|smoke|fluid|realworld>_<pred|gt>.mp4
     4_flow_to_video/       one folder per method, the same <scene>.mp4 in each:
       ours_wanmove/        Wan-Move driven by our tracks
       cogvideox/           720x480 clips had their letterbox bars cropped (-> 720x402);
                            the four Physics-IQ scenes are already 960x540
       physctrl/  physgen/  physgaussian/
       wanmove_notrack/     Wan-Move from the same start frame without trajectories
                            Every method shows the first half of its clip: 2.5 s of 5 s (ours
                            41 of 81 frames or 40 of 80; CogVideoX 25 of 49 or 20 of 40),
                            except CogVideoX coffee_pour, which keeps every other frame of the
                            whole clip (2x speed)
     6_force/               explicit_force<1|2>_<input|pred|gt>.mp4
                            implicit_<input|pred|gt>.mp4
     7_failure/             longhorizon_realworld_<pred|gt>.mp4
                            generalization_fluid_<pred|gt>.mp4
                            highfreq_rigid_<pred|gt>.mp4
   Folder numbers follow the Experiments sections (5 · Occupancy has no clips).
   Originals and full-length versions of processed clips are kept outside the
   site, in ../unused/ (pre_reencode/, pre_trim/, still under their old 3_flow names;
   web_videos/ for the PhysCtrl, PhysGen, PhysGaussian and no-trajectory clips).

   Paths are relative to index.html, e.g. "assets/webvids/3_flow/flow2d_rigid_pred.mp4".

   TIP: keep clips short (2-5s) and compress them (e.g. with ffmpeg:
        ffmpeg -i in.mov -vcodec libx264 -crf 24 -an -movflags +faststart out.mp4)
   ===================================================================== */

window.SITE_CONFIG = {

  /* ---------------------------------------------------------------
     TEASER — the big media block under the title.
     --------------------------------------------------------------- */
  teaser: {
    src: "assets/teaser.png",                       // e.g. "assets/teaser.mp4" or "assets/teaser.png"
    poster: "",                    // optional still image for a video
    caption:
      "<b>Unified implicit 3D physics.</b> Trained only on 2D video latents, Neural Voxel " +
      "Dynamics advects features within a 3D voxel grid — unifying rigid bodies, fluids, and " +
      "smoke in one model. From a few input frames it rolls the voxel latents forward in time, " +
      "and the predicted trajectory can be read out as optical flow or used to drive " +
      "flow-guided video generation.",
  },

  /* ---------------------------------------------------------------
     PIPELINE — the method overview figure.
     --------------------------------------------------------------- */
  pipeline: {
    src: "assets/pipeline.png",                       // e.g. "assets/pipeline.png"
    poster: "",
    caption:
      "<b>Method overview.</b> 2D semantic features are lifted into a 3D latent voxel grid; a " +
      "generative feature advector f<sub>θ</sub> implicitly simulates action-conditioned " +
      "physics via flow matching, supervised by video-derived signals only.",
  },

  /* ---------------------------------------------------------------
     GALLERIES — every video/image grid on the page, keyed by the id
     of its <div class="gallery"> mount in index.html.

     Each item is one tile:
       badge : small pill shown in the corner (dataset / material)
       title : tile heading
       desc  : one-line description
       src   : "" for a labelled PLACEHOLDER, or a path to an image/video
       compareSrc + label/compareLabel: optional second clip with a reveal slider
       comparison: true to show a reveal slider even while comparison clips are pending
       poster / comparePoster: optional stills
       placeholderText: optional message shown while a clip is unavailable
       aspect: "W / H" to size the tile to a non-square clip instead of
               letterboxing it inside a square slot (e.g. "16 / 9")
       grid: [{ src, tag, compareSrc }, ...] to show several clips as one
               card, in a small grid that plays together on hover (e.g. novel
               views); `tag` is the corner pill, and a tile with `compareSrc`
               gets its own reveal slider
       frames: N to show the clip as N sampled stills in a small grid
               instead of a looping video (evenly spaced across the whole
               clip, first and last frame included) — good for very short
               "input" clips that flicker when played
       autoplay: true to start a reveal-slider tile playing (muted, looping)
               as soon as the page loads rather than on hover; skipped for
               visitors who ask for reduced motion
       methods: [{ label, src, selected }, ...] to show one clip per method with a
               row of buttons under it instead of a reveal slider; the tile opens
               on the `selected` method, and a method whose src is "" gets a
               disabled button until its clip is added

     Leave src:"" to keep a clearly-labelled placeholder until you have the
     clip; drop in a path (e.g. "assets/webvids/xyz.mp4") to fill it.
     --------------------------------------------------------------- */
  galleries: {

    /* Hero preview under the TL;DR: two of the section-4 scenes ("flowgen-mount" has the
       other eight), autoplaying so there is motion without a hover */
    "preview-mount": [
      { badge: "Rigid + fluid", title: "Potato in water", autoplay: true, aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/potato_in_water.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/potato_in_water.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/potato_in_water.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/potato_in_water.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/potato_in_water.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/potato_in_water.mp4", selected: true },
        ] },
      { badge: "Smoke", title: "Paper smoke", autoplay: true, aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/paper_smoke.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/paper_smoke.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/paper_smoke.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/paper_smoke.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/paper_smoke.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/paper_smoke.mp4", selected: true },
        ] },
    ],

    /* 1 — Latent prediction: predicted V-JEPA dynamics vs. reference */
    "latent-vids-mount": [
      { badge: "CLEVRER", title: "Rigid-body collisions", desc: "Predicted vs. reference dynamics.",
        label: "Prediction", src: "assets/webvids/1_latent_prediction/qualitative_rigid_pred.mp4",
        compareLabel: "Reference", compareSrc: "assets/webvids/1_latent_prediction/qualitative_rigid_gt.mp4" },
      { badge: "PhysInOne", title: "Fluid pouring", desc: "Fluid poured from a few initial frames.",
        label: "Prediction", src: "assets/webvids/1_latent_prediction/qualitative_fluid_pred.mp4",
        compareLabel: "Reference", compareSrc: "assets/webvids/1_latent_prediction/qualitative_fluid_gt.mp4" },
      { badge: "PhysGaia", title: "Smoke dispersion", desc: "Heterogeneous smoke, single pipeline.",
        label: "Prediction", src: "assets/webvids/1_latent_prediction/qualitative_smoke_pred.mp4",
        compareLabel: "Reference", compareSrc: "assets/webvids/1_latent_prediction/qualitative_smoke_gt.mp4" },
    ],

    /* 1 — Multi-view visualization of the lifted latent */
    "multiview1-mount": [
      { badge: "Multi-view", title: "Multiview input", desc: "Latent lifted from multiple cameras, rendered as a rotating 3D point cloud.",
        src: "assets/webvids/1_latent_prediction/multiview_input_multi.mp4" },
      { badge: "Single-view", title: "Single view input",
        desc: "Lifted from one camera (sample 45), rendered at four novel yaw angles. Hover to play all four together.",
        grid: [
          { tag: "\u221245\u00b0", src: "assets/webvids/1_latent_prediction/multiview_input_single_m45.mp4" },
          { tag: "\u221227\u00b0", src: "assets/webvids/1_latent_prediction/multiview_input_single_m27.mp4" },
          { tag: "+27\u00b0",       src: "assets/webvids/1_latent_prediction/multiview_input_single_p27.mp4" },
          { tag: "+45\u00b0",       src: "assets/webvids/1_latent_prediction/multiview_input_single_p45.mp4" },
        ] },
    ],

    /* 2 — Decoded RGB videos (V-JEPA-to-video head) */
    "rgb-vids-mount": [
      { badge: "Rigid", title: "Decoded rigid-body clip", desc: "Predicted RGB vs. ground truth.",
        label: "Prediction", src: "assets/webvids/2_rgb_decoder/decoded_rigid_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/2_rgb_decoder/decoded_rigid_gt.mp4" },
      { badge: "Smoke", title: "Decoded smoke clip", desc: "Predicted RGB vs. ground truth.",
        label: "Prediction", src: "assets/webvids/2_rgb_decoder/decoded_smoke_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/2_rgb_decoder/decoded_smoke_gt.mp4" },
      { badge: "Occlusion", title: "Temporary occlusion", desc: "Predicted RGB vs. ground truth.",
        label: "Prediction", src: "assets/webvids/2_rgb_decoder/decoded_occlusion_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/2_rgb_decoder/decoded_occlusion_gt.mp4" },
    ],

    /* 2 — Multi-view visualization of decoded RGB (one slot per decoded clip).
       HIDDEN in index.html, so these clips live under unused/ and are not shipped;
       move them back up a level if the tab is restored. */
    "multiview2-mount": [
      { badge: "Rigid", title: "Novel view \u2014 Rigid body",
        desc: "Decoded rigid-body clip at four novel yaw angles. Drag across a view to reveal prediction vs. GT.",
        label: "Pred", compareLabel: "GT",
        grid: [
          { tag: "\u221245\u00b0", src: "assets/webvids/unused/2_rgb_decoder/multiview_rigid_m45_pred.mp4",
            compareSrc: "assets/webvids/unused/2_rgb_decoder/multiview_rigid_m45_gt.mp4" },
          { tag: "\u221227\u00b0", src: "assets/webvids/unused/2_rgb_decoder/multiview_rigid_m27_pred.mp4",
            compareSrc: "assets/webvids/unused/2_rgb_decoder/multiview_rigid_m27_gt.mp4" },
          { tag: "+27\u00b0", src: "assets/webvids/unused/2_rgb_decoder/multiview_rigid_p27_pred.mp4",
            compareSrc: "assets/webvids/unused/2_rgb_decoder/multiview_rigid_p27_gt.mp4" },
          { tag: "+45\u00b0", src: "assets/webvids/unused/2_rgb_decoder/multiview_rigid_p45_pred.mp4",
            compareSrc: "assets/webvids/unused/2_rgb_decoder/multiview_rigid_p45_gt.mp4" },
        ] },
      { badge: "Fluid", title: "Novel view \u2014 Fluid",
        desc: "Decoded fluid clip rendered from four camera viewpoints.",
        grid: [
          { tag: "cam 000", src: "assets/webvids/unused/2_rgb_decoder/multiview_fluid_cam000_pred.mp4" },
          { tag: "cam 001", src: "assets/webvids/unused/2_rgb_decoder/multiview_fluid_cam001_pred.mp4" },
          { tag: "cam 011", src: "assets/webvids/unused/2_rgb_decoder/multiview_fluid_cam011_pred.mp4" },
          { tag: "cam 012", src: "assets/webvids/unused/2_rgb_decoder/multiview_fluid_cam012_pred.mp4" },
        ] },
      { badge: "Smoke", title: "Novel view \u2014 Smoke",
        desc: "Decoded smoke clip at four novel yaw angles.",
        grid: [
          { tag: "\u221220\u00b0", src: "assets/webvids/unused/2_rgb_decoder/multiview_smoke_m20_pred.mp4" },
          { tag: "\u221210\u00b0", src: "assets/webvids/unused/2_rgb_decoder/multiview_smoke_m10_pred.mp4" },
          { tag: "+10\u00b0", src: "assets/webvids/unused/2_rgb_decoder/multiview_smoke_p10_pred.mp4" },
          { tag: "+20\u00b0", src: "assets/webvids/unused/2_rgb_decoder/multiview_smoke_p20_pred.mp4" },
        ] },
    ],

    /* 3a — 2D flow comparisons: HG is prediction; GT is ground truth. */
    "flow-mount": [
      { badge: "Rigid", title: "Rigid-body flow", desc: "Predicted vs. reference 2D flow.",
        label: "Prediction", src: "assets/webvids/3_flow/flow2d_rigid_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/3_flow/flow2d_rigid_gt.mp4" },
      { badge: "Smoke", title: "Smoke flow", desc: "Predicted vs. reference 2D flow.",
        label: "Prediction", src: "assets/webvids/3_flow/flow2d_smoke_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/3_flow/flow2d_smoke_gt.mp4" },
      { badge: "Fluid", title: "Fluid flow", desc: "Predicted vs. reference 2D flow.",
        label: "Prediction", src: "assets/webvids/3_flow/flow2d_fluid_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/3_flow/flow2d_fluid_gt.mp4" },
    ],

    /* 3a — the real-world example, on a row of its own so the dense
       trajectory streaks stay readable */
    "flow-real-mount": [
      { badge: "Real world", title: "Real world mixed dynamics",
        desc: "Predicted vs. reference 2D flow on real footage with mixed materials.",
        aspect: "854 / 480",
        label: "Prediction", src: "assets/webvids/3_flow/flow2d_realworld_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/3_flow/flow2d_realworld_gt.mp4" },
    ],

    /* 4 — estimated 2D flow driving Wan-Move (i2v + 2D trajectories) on real footage, one
       clip per method behind a row of buttons. To add a baseline clip, drop it in
       4_flow_to_video/<method>/<scene>.mp4 and put that path in the matching src. */
    "flowgen-mount": [
      { badge: "Fluid", title: "Coffee pour", desc: "Pouring into a cup. CogVideoX shown at 2× speed.", aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/coffee_pour.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/coffee_pour.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/coffee_pour.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/coffee_pour.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/coffee_pour.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/coffee_pour.mp4", selected: true },
        ] },
      { badge: "Rigid + fluid", title: "Fruit dropped in water", desc: "Impact and splash.", aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/fruit_drop.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/fruit_drop.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/fruit_drop.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/fruit_drop.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/fruit_drop.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/fruit_drop.mp4", selected: true },
        ] },
      { badge: "Rigid + fluid", title: "Stone dropped in water", desc: "Expanding surface ripples.", aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/stone_drop.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/stone_drop.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/stone_drop.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/stone_drop.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/stone_drop.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/stone_drop.mp4", selected: true },
        ] },
      { badge: "Fluid", title: "Liquid on a duck", desc: "Liquid poured over a floating toy.", aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/liquid_on_duck.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/liquid_on_duck.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/liquid_on_duck.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/liquid_on_duck.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/liquid_on_duck.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/liquid_on_duck.mp4", selected: true },
        ] },
      { badge: "Fluid", title: "Water drops", desc: "Droplets on a wet surface.", aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/water_drops.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/water_drops.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/water_drops.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/water_drops.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/water_drops.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/water_drops.mp4", selected: true },
        ] },
      { badge: "Fluid", title: "Water waves", desc: "Turbulent flow over rocks.", aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/water_waves.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/water_waves.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/water_waves.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/water_waves.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/water_waves.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/water_waves.mp4", selected: true },
        ] },
      { badge: "Smoke", title: "Volcano", desc: "Eruption plume over flowing lava.", aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/volcano.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/volcano.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/volcano.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/volcano.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/volcano.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/volcano.mp4", selected: true },
        ] },
      { badge: "Deformable", title: "Weight on a pillow", desc: "Soft-body contact and recovery.", aspect: "16 / 9",
        methods: [
          { label: "CogVideoX",                  src: "assets/webvids/4_flow_to_video/cogvideox/weight_on_pillow.mp4" },
          { label: "PhysCtrl",                   src: "assets/webvids/4_flow_to_video/physctrl/weight_on_pillow.mp4" },
          { label: "PhysGen",                    src: "assets/webvids/4_flow_to_video/physgen/weight_on_pillow.mp4" },
          { label: "PhysGaussian",               src: "assets/webvids/4_flow_to_video/physgaussian/weight_on_pillow.mp4" },
          { label: "Wan-Move (no trajectories)", src: "assets/webvids/4_flow_to_video/wanmove_notrack/weight_on_pillow.mp4" },
          { label: "Our tracks + Wan-Move",      src: "assets/webvids/4_flow_to_video/ours_wanmove/weight_on_pillow.mp4", selected: true },
        ] },
    ],

    /* 6a — Two rows: input on the left, prediction (HG) vs. GT on the right. */
    "explicit-force-mount": [
      { badge: "Force 1", title: "Input", desc: "Input frames: initial scene and applied force.",
        frames: 4, src: "assets/webvids/6_force/explicit_force1_input.mp4" },
      { badge: "Force 1", title: "Prediction", desc: "Predicted dynamics vs. ground truth.",
        label: "Prediction", src: "assets/webvids/6_force/explicit_force1_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/6_force/explicit_force1_gt.mp4" },
      { badge: "Force 2", title: "Input", desc: "Input frames: initial scene and applied force.",
        frames: 4, src: "assets/webvids/6_force/explicit_force2_input.mp4" },
      { badge: "Force 2", title: "Prediction", desc: "Predicted dynamics vs. ground truth.",
        label: "Prediction", src: "assets/webvids/6_force/explicit_force2_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/6_force/explicit_force2_gt.mp4" },
    ],

    /* 6b — Implicit forces inferred from video: input frames, then pred vs. GT */
    "implicit-force-mount": [
      { badge: "Implicit", title: "Input", desc: "Input frames: no explicit force is given.",
        aspect: "640 / 720", frames: 4, src: "assets/webvids/6_force/implicit_input.mp4" },
      { badge: "Implicit", title: "Prediction", desc: "Force inferred from motion cues, vs. ground truth.",
        aspect: "640 / 720",
        label: "Prediction", src: "assets/webvids/6_force/implicit_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/6_force/implicit_gt.mp4" },
    ],

    /* 7 — Failure examples: the 16:9 long-horizon clip gets its own wide row */
    "failure-long-mount": [
      { badge: "Failure", title: "Long-horizon drift",
        desc: "The full 5 s real-world clip: flow tracks the impact, then degrades as the rollout runs on.",
        aspect: "854 / 480",
        label: "Prediction", src: "assets/webvids/7_failure/longhorizon_realworld_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/7_failure/longhorizon_realworld_gt.mp4" },
    ],

    "failure-mount": [
      { badge: "Failure", title: "Decoder and predictor generalization",
        desc: "Decoded fluid clip: neither the decoder nor the predictor generalizes to this scene.",
        label: "Prediction", src: "assets/webvids/7_failure/generalization_fluid_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/7_failure/generalization_fluid_gt.mp4" },
      { badge: "Failure", title: "High-frequency detail",
        desc: "The earlier decoded rigid-body clip: fine deformation is lost at the current voxel size.",
        label: "Prediction", src: "assets/webvids/7_failure/highfreq_rigid_pred.mp4",
        compareLabel: "GT", compareSrc: "assets/webvids/7_failure/highfreq_rigid_gt.mp4" },
    ],

  },
};
