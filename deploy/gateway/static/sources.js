// The Sources panel: what the tester opened from a citation in the appliance, delivered to this browser.
//
// Self-contained on purpose: it needs only an element tree and the /api/sources routes, not noVNC, so a future web
// frontend can use it as is. Everything shown here (excerpts, labels) is client material, so it is only ever set
// with textContent, and URLs are only ever built from an offer id the server minted.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const KIND_LABEL = { video: "Video", audio: "Audio", pdf: "PDF", text: "Text", document: "Document", image: "Image" };
  const MAX_RENDERED_LINES = 5000;
  const CONTEXT_LINES = 200;
  const offers = [];          // newest first
  let current = null;

  // ---------- helpers ----------

  function el(tag, attrs, text) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs || {})) node.setAttribute(name, value);
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function url(offer, variant) {
    return `/api/sources/${encodeURIComponent(offer.id)}/${encodeURIComponent(variant)}`;
  }

  function has(offer, variant) {
    return offer.variants.includes(variant);
  }

  function clock(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    const pad = (n) => String(n).padStart(2, "0");
    // As the app writes it: hours only when there are any
    const minutes = `${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
    return s >= 3600 ? `${Math.floor(s / 3600)}:${minutes}` : minutes;
  }

  function where(offer) {
    if (offer.location) return offer.location;
    if (offer.kind === "video" || offer.kind === "audio") return `${clock(offer.start_sec)}–${clock(offer.end_sec)}`;
    return "";
  }

  function size(bytes) {
    if (!bytes) return "";
    if (bytes < 1048576) return ` (${Math.max(1, Math.round(bytes / 1024))} KB)`;
    return ` (${(bytes / 1048576).toFixed(bytes < 10485760 ? 1 : 0)} MB)`;
  }

  function remember(name, value) {
    try { localStorage.setItem(name, value); } catch (e) { /* private window: not remembered */ }
  }

  function recall(name) {
    try { return localStorage.getItem(name); } catch (e) { return null; }
  }

  // ---------- players ----------

  function mediaPlayer(offer, src, startAt, autoplay) {
    const tag = offer.kind === "video" ? "video" : "audio";
    const player = el(tag, { controls: "", preload: "metadata", playsinline: "" });
    const status = el("p", { class: "note" }, "Preparing… a recording the browser can't play as it is gets converted first.");
    const failed = el("p", { class: "error", hidden: "" },
      "The browser couldn't play this source. Download it below to play it in your own player.");
    player.addEventListener("loadedmetadata", () => {
      status.hidden = true;
      if (startAt > 0 && player.currentTime < startAt - 0.5) player.currentTime = startAt;
    }, { once: true });
    player.addEventListener("error", () => { status.hidden = true; failed.hidden = false; });
    player.src = src;
    if (autoplay) player.play().catch(() => { /* the browser wants a click first; the controls are there */ });
    return [player, status, failed];
  }

  function recording(offer, autoplay) {
    const box = el("div", { class: "viewer" });
    const stage = el("div");
    box.append(stage);
    if (!has(offer, "excerpt")) {
      // A per-chunk clip already is the cited span
      stage.append(...mediaPlayer(offer, url(offer, "media"), 0, autoplay));
      return box;
    }
    const passage = el("button", { type: "button", "aria-pressed": "true" }, "Cited passage");
    const whole = el("button", { type: "button", "aria-pressed": "false" }, "Whole recording");
    const caption = el("p", { class: "note" });
    const show = (which, play) => {
      stage.replaceChildren();
      passage.setAttribute("aria-pressed", String(which === "excerpt"));
      whole.setAttribute("aria-pressed", String(which === "media"));
      if (which === "excerpt") {
        caption.textContent = `Plays ${clock(offer.start_sec)}–${clock(offer.end_sec)} with 5 seconds either side.`;
        stage.append(...mediaPlayer(offer, url(offer, "excerpt"), 0, play));
      } else {
        caption.textContent = `Starts at the cited moment, ${clock(offer.start_sec)}. Seek anywhere in the recording.`;
        stage.append(...mediaPlayer(offer, `${url(offer, "media")}#t=${offer.start_sec}`, offer.start_sec, play));
      }
    };
    passage.addEventListener("click", () => show("excerpt", true));
    whole.addEventListener("click", () => show("media", true));
    box.prepend(el("div", { class: "row" }), caption);
    box.firstChild.append(passage, whole);
    show("excerpt", autoplay);
    return box;
  }

  function pdf(offer) {
    const box = el("div", { class: "viewer" });
    const page = offer.page_start || 1;
    const frame = el("iframe", { class: "pdf", title: `${offer.doc_label} PDF`, src: `${url(offer, "media")}#page=${page}` });
    const open = el("a", { class: "button", href: `${url(offer, "media")}#page=${page}`, target: "_blank",
                           rel: "noopener" }, "Open in a new tab");
    box.append(frame, el("div", { class: "row" }));
    box.lastChild.append(open);
    return box;
  }

  function textLines(offer) {
    const box = el("div", { class: "viewer" });
    const status = el("p", { class: "note" }, "Loading…");
    box.append(status);
    if (!has(offer, "text")) {
      status.textContent = "This file is too large to show here. Download it below.";
      return box;
    }
    const first = offer.page_start || 1;
    const last = offer.page_end || first;
    fetch(url(offer, "text"), { credentials: "same-origin" })
      .then((r) => { if (!r.ok) throw new Error(r.statusText); return r.text(); })
      .then((text) => {
        const lines = text.split("\n");
        if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();  // the file's final line break
        let from = 1, to = lines.length;
        if (lines.length > MAX_RENDERED_LINES) {
          from = Math.max(1, first - CONTEXT_LINES);
          to = Math.min(lines.length, last + CONTEXT_LINES);
          status.textContent = `Showing lines ${from}–${to} of ${lines.length}. Download for the whole file.`;
        } else {
          status.textContent = first === last ? `Line ${first} is highlighted.` : `Lines ${first}–${last} are highlighted.`;
        }
        const list = el("div", { class: "lines" });
        let target = null;
        for (let n = from; n <= to; n++) {
          const row = el("div");
          if (n >= first && n <= last) { row.className = "cited"; target = target || row; }
          row.append(el("span", { class: "n" }, String(n)), el("span", {}, lines[n - 1]));
          list.append(row);
        }
        box.append(list);
        if (target) list.scrollTop = Math.max(0, target.offsetTop - list.clientHeight / 3);
      })
      .catch(() => { status.className = "error"; status.textContent = "This file couldn't be loaded. Download it below."; });
    return box;
  }

  function image(offer) {
    const box = el("div", { class: "viewer" });
    const failed = el("p", { class: "error", hidden: "" }, "This image couldn't be shown. Download it below.");
    const img = el("img", { src: url(offer, "media"), alt: offer.doc_label || "Cited image" });
    img.addEventListener("error", () => { img.hidden = true; failed.hidden = false; });
    box.append(img, failed);
    return box;
  }

  function viewer(offer, autoplay) {
    switch (offer.kind) {
      case "video":
      case "audio": return recording(offer, autoplay);
      case "pdf": return pdf(offer);
      case "text": return textLines(offer);
      case "image": return image(offer);
      default: return el("p", { class: "note" },
        "Browsers can't open this kind of document. The cited passage is below, word for word; download the file to read it in full.");
    }
  }

  // ---------- panel ----------

  function show(offer, autoplay) {
    current = offer;
    const box = $("current");
    box.replaceChildren();
    box.hidden = false;
    $("empty").hidden = true;

    box.append(el("div", { class: "title" }, `${offer.doc_label || "Source"} · ${KIND_LABEL[offer.kind] || ""}`));
    const place = where(offer);
    if (place) box.append(el("div", { class: "where" }, place));
    box.append(viewer(offer, autoplay));

    const actions = el("div", { class: "row" });
    actions.append(el("a", { class: "button primary", href: url(offer, "original"), download: offer.download_name },
      `Download original${size(offer.size_bytes)}`));
    if (has(offer, "excerpt")) {
      const name = `${offer.download_name.replace(/\.[^.]+$/, "")} (cited passage).${offer.kind === "video" ? "mp4" : "m4a"}`;
      actions.append(el("a", { class: "button", href: url(offer, "excerpt"), download: name }, "Download cited passage"));
    }
    box.append(actions);

    if (offer.excerpt) {
      if (offer.speakers && offer.speakers.length) box.append(el("div", { class: "where" }, offer.speakers.join(", ")));
      box.append(el("div", { class: "excerpt" }, offer.excerpt));
    }
    renderHistory();
  }

  function renderHistory() {
    const list = $("history");
    list.replaceChildren();
    const earlier = offers.filter((o) => o !== current);
    $("history-heading").hidden = earlier.length === 0;
    for (const offer of earlier) {
      const item = el("li", { tabindex: "0", role: "button" });
      const text = el("div");
      text.append(el("div", { class: "title" }, offer.doc_label || "Source"), el("div", { class: "where" }, where(offer)));
      item.append(el("span", { class: "kind" }, KIND_LABEL[offer.kind] || ""), text);
      const pick = () => show(offer, false);
      item.addEventListener("click", pick);
      item.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      list.append(item);
    }
  }

  let unseen = 0;
  function setCollapsed(collapsed) {
    document.body.classList.toggle("collapsed", collapsed);
    $("toggle").setAttribute("aria-pressed", String(!collapsed));
    remember("onyx.sources.collapsed", collapsed ? "1" : "0");
    if (!collapsed) { unseen = 0; $("unseen").hidden = true; }
  }

  $("toggle").addEventListener("click", () => setCollapsed(!document.body.classList.contains("collapsed")));
  setCollapsed(recall("onyx.sources.collapsed") === "1");

  // ---------- feed ----------

  const feed = new EventSource("/api/sources/stream");
  feed.addEventListener("source", (event) => {
    const offer = JSON.parse(event.data);
    if (offers.some((o) => o.id === offer.id)) return;
    offers.unshift(offer);
    $("feed-error").hidden = true;
    if (offer.live) {
      // A click in the appliance just now: bring it into view and start it
      if (document.body.classList.contains("collapsed")) {
        unseen += 1;
        $("unseen").textContent = String(unseen);
        $("unseen").hidden = false;
      }
      show(offer, true);
    } else {
      renderHistory();
    }
  });
  feed.addEventListener("ready", () => {
    if (!current && offers.length) show(offers[0], false);
  });
  feed.addEventListener("error", () => {
    if (feed.readyState === EventSource.CLOSED) {
      $("feed-error").textContent = "Sources stopped arriving. You may have been signed out; reload the page.";
      $("feed-error").hidden = false;
    }
  });
})();
