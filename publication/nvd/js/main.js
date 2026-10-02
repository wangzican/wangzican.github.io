/* =====================================================================
   main.js — renders media + galleries from window.SITE_CONFIG.
   You normally do NOT need to edit this file; edit js/config.js instead.
   ===================================================================== */
(function () {
  "use strict";

  var VIDEO_EXT = /\.(mp4|webm|ogg|mov)$/i;
  var cfg = window.SITE_CONFIG || {};
  var REDUCED_MOTION = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  /* Tiles marked `autoplay` start playing on load instead of on hover,
     unless the visitor has asked for reduced motion. */
  function autoplays(item) {
    return !!item.autoplay && !REDUCED_MOTION;
  }

  function setupVideo(v, placeholderLabel) {
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = "metadata";
    v.setAttribute("aria-label", placeholderLabel || "result video");
  }

  function toggleVideo(v) {
    v.paused ? v.play().catch(function () {}) : v.pause();
  }

  /* Build a media element: <video>, <img>, or a labelled placeholder. */
  function media(item, placeholderLabel, attachVideoControls) {
    item = item || {};
    if (attachVideoControls !== false) attachVideoControls = true;
    var src = (item.src || "").trim();

    if (!src) {
      var ph = document.createElement("div");
      ph.className = "media-ph";
      ph.innerHTML =
        '<span class="media-ph__label"><b>' +
        (placeholderLabel || "Placeholder") +
        "</b><br>" + (item.placeholderText || "set <code>src</code> in js/config.js") + "</span>";
      return ph;
    }

    if (VIDEO_EXT.test(src)) {
      var v = document.createElement("video");
      v.className = "media";
      v.src = src;
      if (item.poster) v.poster = item.poster;
      setupVideo(v, placeholderLabel);
      // Hover (desktop) / tap (mobile) to play.
      if (attachVideoControls) {
        v.addEventListener("mouseenter", function () { v.play().catch(function () {}); });
        v.addEventListener("mouseleave", function () { v.pause(); });
        v.addEventListener("click", function () { toggleVideo(v); });
      }
      return v;
    }

    var img = document.createElement("img");
    img.className = "media";
    img.src = src;
    img.alt = placeholderLabel || "figure";
    img.loading = "lazy";
    return img;
  }

  function comparisonMedia(item, externalPlayback) {
    var compareSrc = (item.compareSrc || "").trim();
    if (!compareSrc && !item.comparison) return media(item, item.title || "Placeholder");

    var wrap = document.createElement("div");
    wrap.className = "compare";
    wrap.style.setProperty("--split", "50%");
    wrap.setAttribute("role", "slider");
    wrap.setAttribute("aria-label", (item.title || "Video") + " comparison reveal");
    wrap.setAttribute("aria-valuemin", "0");
    wrap.setAttribute("aria-valuemax", "100");
    wrap.setAttribute("aria-valuenow", "50");
    wrap.tabIndex = 0;

    var bottom = document.createElement("div");
    bottom.className = "compare__layer compare__layer--bottom";
    bottom.appendChild(media({ src: compareSrc, poster: item.comparePoster, placeholderText: item.placeholderText }, item.compareLabel || "comparison video", false));

    var top = document.createElement("div");
    top.className = "compare__layer compare__layer--top";
    top.appendChild(media(item, item.label || item.title || "result video", false));

    var handle = document.createElement("div");
    handle.className = "compare__handle";
    handle.setAttribute("aria-hidden", "true");
    handle.innerHTML = '<span class="compare__knob"></span>';

    // GT sits on the left, prediction on the right (see the top layer's clip-path).
    // Both labels share one row, so a long one can use the room a short one leaves.
    var labels = document.createElement("div");
    labels.className = "compare__labels";

    var leftLabel = document.createElement("span");
    leftLabel.className = "compare__label compare__label--left";
    leftLabel.textContent = item.compareLabel || "Reference";

    var rightLabel = document.createElement("span");
    rightLabel.className = "compare__label compare__label--right";
    rightLabel.textContent = item.label || "Prediction";

    labels.appendChild(leftLabel);
    labels.appendChild(rightLabel);

    wrap.appendChild(bottom);
    wrap.appendChild(top);
    wrap.appendChild(handle);
    wrap.appendChild(labels);

    function videos() {
      return Array.prototype.slice.call(wrap.querySelectorAll("video")).filter(function (v) {
        return v.getAttribute("src");
      });
    }

    function useSingleVideo() {
      var comparisonVideo = bottom.querySelector("video");
      wrap.classList.add("compare--single");
      wrap.removeAttribute("role");
      wrap.removeAttribute("aria-valuemin");
      wrap.removeAttribute("aria-valuemax");
      wrap.removeAttribute("aria-valuenow");
      if (comparisonVideo) {
        comparisonVideo.pause();
        comparisonVideo.removeAttribute("src");
        comparisonVideo.load();
      }
    }

    var isPlaying = false;

    function setPlaying(play) {
      isPlaying = play;
      videos().forEach(function (v) {
        if (!play) v.pause();
        else if (!v.ended) v.play().catch(function () {});   // one that ended early waits (see syncLoops)
      });
    }

    function setSplit(percent) {
      percent = Math.max(0, Math.min(100, percent));
      wrap.style.setProperty("--split", percent + "%");
      wrap.setAttribute("aria-valuenow", String(Math.round(percent)));
    }

    function updateFromPointer(e) {
      var rect = wrap.getBoundingClientRect();
      setSplit(((e.clientX - rect.left) / rect.width) * 100);
    }

    function checkRatios() {
      var allVideos = videos();
      if (allVideos.length < 2) return;
      if (!allVideos[0].videoWidth || !allVideos[1].videoWidth) return;

      var topRatio = allVideos[0].videoWidth / allVideos[0].videoHeight;
      var bottomRatio = allVideos[1].videoWidth / allVideos[1].videoHeight;
      if (Math.abs(topRatio - bottomRatio) > 0.01) useSingleVideo();
    }

    /* Clips of different lengths (e.g. two generators with their own frame
       counts and rates) drift apart if each loops on its own, so loop them
       together instead: the shorter one holds its last frame until the longer
       one ends, then both restart. Equal-length pairs keep the native loop. */
    function syncLoops() {
      var allVideos = videos();
      if (allVideos.length < 2) return;
      if (!allVideos[0].duration || !allVideos[1].duration) return;
      if (Math.abs(allVideos[0].duration - allVideos[1].duration) < 0.05) return;
      allVideos.forEach(function (v) { v.loop = false; });
    }

    function restartWhenAllEnded() {
      var allVideos = videos();
      if (!allVideos.every(function (v) { return v.ended; })) return;
      allVideos.forEach(function (v) {
        v.currentTime = 0;
        if (isPlaying) v.play().catch(function () {});
      });
    }

    videos().forEach(function (v) {
      v.addEventListener("loadedmetadata", function () {
        checkRatios();
        syncLoops();
      });
      v.addEventListener("ended", restartWhenAllEnded);
    });

    wrap.setPlaying = setPlaying;   // so a tile grid can run several in step

    // an autoplaying tile keeps running when the pointer leaves; hover and tap
    // still start it if the browser blocked the autoplay
    var autoplay = autoplays(item);

    wrap.addEventListener("pointerenter", function (e) {
      updateFromPointer(e);
      if (!externalPlayback && e.pointerType !== "touch") setPlaying(true);
    });
    wrap.addEventListener("pointermove", updateFromPointer);
    wrap.addEventListener("pointerleave", function (e) {
      if (!externalPlayback && !autoplay && e.pointerType !== "touch") setPlaying(false);
    });
    wrap.addEventListener("pointerdown", function (e) {
      updateFromPointer(e);
      if (!externalPlayback && e.pointerType === "touch") setPlaying(autoplay || !isPlaying);
    });
    wrap.addEventListener("keydown", function (e) {
      var current = parseFloat(wrap.getAttribute("aria-valuenow") || "50");
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setSplit(current - 5);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setSplit(current + 5);
      }
    });

    return wrap;
  }

  /* Several methods on one scene: one clip at a time, picked with a row of
     buttons under it. A method with no clip yet gets a disabled button. */
  function methodSwitch(item) {
    var methods = item.methods || [];
    var autoplay = autoplays(item);

    var stage = document.createElement("div");
    stage.className = "mswitch";

    var bar = document.createElement("div");
    bar.className = "mswitch__bar";
    bar.setAttribute("role", "group");
    bar.setAttribute("aria-label", (item.title || "Clip") + ": method");

    var clips = [];       // one element per method, built the first time it is shown
    var buttons = [];
    var current = -1;
    var isPlaying = false;

    function hasClip(m) { return !!(m.src || "").trim(); }

    function setPlaying(play) {
      isPlaying = play;
      var clip = clips[current];
      if (!clip || clip.tagName !== "VIDEO") return;
      if (play) clip.play().catch(function () {});
      else clip.pause();
    }

    /* Each method's clip starts from its first frame when it is picked. */
    function show(i) {
      if (i === current) return;
      var prev = clips[current];
      if (prev) {
        if (prev.tagName === "VIDEO") prev.pause();
        prev.hidden = true;
      }
      if (!clips[i]) {
        clips[i] = media(methods[i], methods[i].label, false);
        clips[i].classList.add("mswitch__clip");
        stage.appendChild(clips[i]);
      } else if (clips[i].tagName === "VIDEO") {
        try { clips[i].currentTime = 0; } catch (err) { /* not seekable yet */ }
      }
      clips[i].hidden = false;
      buttons.forEach(function (b, j) {
        b.classList.toggle("active", j === i);
        b.setAttribute("aria-pressed", String(j === i));
      });
      current = i;
      if (isPlaying) setPlaying(true);
    }

    methods.forEach(function (m, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "mswitch__btn";
      b.textContent = m.label || "Method " + (i + 1);
      b.setAttribute("aria-pressed", "false");
      if (hasClip(m)) {
        b.addEventListener("click", function () { show(i); setPlaying(true); });
      } else {
        b.disabled = true;
        b.title = "Clip not added yet";
      }
      bar.appendChild(b);
      buttons.push(b);
    });

    var first = -1;
    methods.forEach(function (m, i) {
      if (hasClip(m) && (first < 0 || m.selected)) first = i;
    });
    if (first >= 0) show(first);
    else stage.appendChild(media({ placeholderText: item.placeholderText }, item.title || "Placeholder"));

    // hovering the clip or its buttons plays it; an autoplaying tile keeps running
    function inside(node) { return !!node && (stage.contains(node) || bar.contains(node)); }
    [stage, bar].forEach(function (el) {
      el.addEventListener("pointerenter", function (e) {
        if (e.pointerType !== "touch") setPlaying(true);
      });
      el.addEventListener("pointerleave", function (e) {
        if (e.pointerType !== "touch" && !autoplay && !inside(e.relatedTarget)) setPlaying(false);
      });
    });
    stage.addEventListener("pointerdown", function (e) {
      if (e.pointerType === "touch") setPlaying(autoplay || !isPlaying);
    });

    stage.setPlaying = setPlaying;
    stage.switchBar = bar;      // mounted under the clip by mountGallery
    return stage;
  }

  /* Show several clips as one card: a small grid that plays together. */
  function videoGrid(item) {
    var tiles = item.grid || [];
    var cols = Math.max(1, Math.ceil(Math.sqrt(tiles.length)));

    var grid = document.createElement("div");
    grid.className = "tilegrid";
    grid.style.setProperty("--cols", String(cols));

    var vids = [];        // plain clips
    var players = [];     // reveal comparisons, which play their own two clips

    tiles.forEach(function (tile) {
      var cell = document.createElement("div");
      cell.className = "tilegrid__cell";
      var compareSrc = (tile.compareSrc || "").trim();

      if (compareSrc) {
        // inherit the card's reveal labels unless the tile overrides them
        var cmp = comparisonMedia({
          src: tile.src, compareSrc: compareSrc,
          poster: tile.poster, comparePoster: tile.comparePoster,
          title: (item.title || "") + " " + (tile.tag || ""),
          label: tile.label || item.label,
          compareLabel: tile.compareLabel || item.compareLabel,
        }, true);
        cell.classList.add("tilegrid__cell--compare");
        cell.appendChild(cmp);
        players.push(cmp);
      } else if ((tile.src || "").trim()) {
        var el = media(tile, tile.tag || item.title || "result video", false);
        el.classList.add("tilegrid__media");
        cell.appendChild(el);
        if (el.tagName === "VIDEO") vids.push(el);
      } else {
        cell.classList.add("tilegrid__cell--empty");
      }

      if (tile.tag) {
        var tag = document.createElement("span");
        tag.className = "tilegrid__label";
        cell.appendChild(tag);
        tag.textContent = tile.tag;
      }
      grid.appendChild(cell);
    });

    if (!vids.length && !players.length) return grid;

    var isPlaying = false;

    function setPlaying(play) {
      isPlaying = play;
      if (play) {                     // keep the views in step with each other
        Array.prototype.slice.call(grid.querySelectorAll("video")).forEach(function (v) {
          try { v.currentTime = 0; } catch (err) { /* not seekable yet */ }
        });
      }
      vids.forEach(function (v) {
        if (play) v.play().catch(function () {});
        else v.pause();
      });
      players.forEach(function (p) { p.setPlaying(play); });
    }

    grid.addEventListener("pointerenter", function (e) {
      if (e.pointerType !== "touch") setPlaying(true);
    });
    grid.addEventListener("pointerleave", function (e) {
      if (e.pointerType !== "touch") setPlaying(false);
    });
    grid.addEventListener("pointerdown", function (e) {
      if (e.pointerType === "touch") setPlaying(!isPlaying);
    });

    return grid;
  }

  /* Show a clip as a small grid of stills instead of a looping video.
     Frames are sampled evenly across the whole clip (first and last
     included), so a clip longer than `count` frames is skipped through
     rather than truncated. */
  function frameStrip(item) {
    var count = Math.max(1, parseInt(item.frames, 10) || 4);
    var cols = Math.ceil(Math.sqrt(count));

    var strip = document.createElement("div");
    strip.className = "filmstrip";
    strip.style.setProperty("--cols", String(cols));

    var cells = [];
    for (var i = 0; i < count; i++) {
      var cell = document.createElement("div");
      cell.className = "filmstrip__cell filmstrip__cell--empty";
      var idx = document.createElement("span");
      idx.className = "filmstrip__idx";
      idx.textContent = String(i + 1);
      cell.appendChild(idx);
      strip.appendChild(cell);
      cells.push(cell);
    }

    var src = (item.src || "").trim();
    if (!src || !VIDEO_EXT.test(src)) return strip;   // keep dashed placeholders

    var probe = document.createElement("video");
    probe.className = "filmstrip__probe";
    probe.src = src;
    probe.setAttribute("aria-hidden", "true");
    setupVideo(probe, item.title || "input frames");
    probe.loop = false;
    probe.preload = "auto";          // must outrank setupVideo's "metadata"
    strip.appendChild(probe);

    /* If the clip cannot be sampled, fall back to the plain hover-to-play video. */
    var settled = false;
    function fallback() {
      if (settled) return;
      settled = true;
      while (strip.firstChild) strip.removeChild(strip.firstChild);
      strip.classList.add("filmstrip--fallback");
      strip.appendChild(media(item, item.title || "input clip"));
    }
    var giveUp = setTimeout(fallback, 8000);

    probe.addEventListener("error", fallback);
    probe.addEventListener("loadeddata", function () {
      if (settled) return;
      var duration = probe.duration;
      var w = probe.videoWidth, h = probe.videoHeight;
      if (!isFinite(duration) || duration <= 0 || !w || !h) { fallback(); return; }
      settled = true;
      clearTimeout(giveUp);

      var last = Math.max(0, duration - 0.001);
      var canvases = cells.map(function (cell) {
        var c = document.createElement("canvas");
        c.className = "filmstrip__frame";
        c.width = w;
        c.height = h;
        cell.insertBefore(c, cell.firstChild);
        return c;
      });

      var at = 0, timer = null;

      function draw() {
        try {
          canvases[at].getContext("2d").drawImage(probe, 0, 0, w, h);
          cells[at].classList.remove("filmstrip__cell--empty");
        } catch (err) { /* frame unavailable — leave the slot blank */ }
      }

      function step() {
        clearTimeout(timer);
        if (at >= count) {          // done: release the decoder
          probe.removeAttribute("src");
          probe.load();
          if (probe.parentNode) probe.parentNode.removeChild(probe);
          return;
        }
        var t = count === 1 ? 0 : Math.min(last, (duration * at) / (count - 1));
        timer = setTimeout(advance, 3000);   // a stuck seek must not stall the rest
        if (Math.abs(probe.currentTime - t) < 1e-4) setTimeout(advance, 0);
        else probe.currentTime = t;
      }

      function advance() {
        clearTimeout(timer);
        draw();
        at++;
        step();
      }

      probe.addEventListener("seeked", advance);
      step();
    });

    return strip;
  }

  /* Mount a single captioned figure (teaser / pipeline). */
  function mountFigure(mountId, item, fallbackLabel) {
    var mount = document.getElementById(mountId);
    if (!mount || !item) return;
    mount.appendChild(media(item, fallbackLabel));
    if (item.caption) {
      var cap = document.createElement("figcaption");
      cap.className = "cap";
      cap.innerHTML = item.caption;
      mount.appendChild(cap);
    }
  }

  /* Mount a gallery of cards. */
  function mountGallery(mountId, items) {
    var mount = document.getElementById(mountId);
    if (!mount || !items) return;
    items.forEach(function (item) {
      var card = document.createElement("figure");
      card.className = "card";

      var mediaWrap = document.createElement("div");
      mediaWrap.className = "card__media";
      // match the slot to the clip so a non-square video leaves no blank bands
      if (item.aspect) mediaWrap.style.setProperty("--media-aspect", item.aspect);
      if (item.badge) {
        var badge = document.createElement("span");
        badge.className = "card__badge";
        badge.textContent = item.badge;
        mediaWrap.appendChild(badge);
      }
      var mediaEl =
        item.methods ? methodSwitch(item) :
        item.grid ? videoGrid(item) :
        item.frames ? frameStrip(item) : comparisonMedia(item);
      mediaWrap.appendChild(mediaEl);
      card.appendChild(mediaWrap);
      if (mediaEl.switchBar) card.appendChild(mediaEl.switchBar);

      var body = document.createElement("figcaption");
      body.className = "card__body";
      body.innerHTML =
        '<h3 class="card__title">' + (item.title || "") + "</h3>" +
        '<p class="card__desc">' + (item.desc || "") + "</p>";
      card.appendChild(body);

      mount.appendChild(card);
      // start an autoplaying reveal only once it is in the page
      if (autoplays(item) && mediaEl.setPlaying) mediaEl.setPlaying(true);
    });
  }

  /* ---- Render everything ---- */
  mountFigure("teaser-mount", cfg.teaser, "Teaser image / video");
  mountFigure("pipeline-mount", cfg.pipeline, "Pipeline figure");

  /* Keyed galleries: mount id -> items (see cfg.galleries in config.js) */
  if (cfg.galleries) {
    Object.keys(cfg.galleries).forEach(function (mountId) {
      mountGallery(mountId, cfg.galleries[mountId]);
    });
  }
  /* Back-compat with the older flat arrays, if present */
  mountGallery("results-mount", cfg.results);
  mountGallery("downstream-mount", cfg.downstream);

  /* ---- Mobile nav toggle ---- */
  var toggle = document.querySelector(".nav__toggle");
  var links = document.querySelector(".nav__links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      var open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    links.addEventListener("click", function (e) {
      if (e.target.tagName === "A") {
        links.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  /* ---- BibTeX copy button ---- */
  var copyBtn = document.querySelector(".bibtex__copy");
  if (copyBtn) {
    copyBtn.addEventListener("click", function () {
      var code = document.querySelector(".bibtex code");
      if (!code) return;
      navigator.clipboard.writeText(code.innerText).then(function () {
        var prev = copyBtn.textContent;
        copyBtn.textContent = "Copied!";
        setTimeout(function () { copyBtn.textContent = prev; }, 1500);
      });
    });
  }
})();
