/* TCR video gallery. Dependency-free; media is fetched only when played. */
"use strict";

(() => {
  const PAGE_SIZE = 12;
  const METHODS = {
    tcr: "TCR (ours)", experts: "Experts", featcal: "FeatCal", regmeanpp: "RegMean++",
    soup: "Mean Soup", ties: "TIES", knots_ties: "KnOTS-TIES", regmean: "RegMean",
    task_arithmetic: "Task Arithmetic", wudi: "WUDI"
  };
  const TASKS = {
    task1: "Task 1", task2: "Task 2", libero_spatial: "LIBERO-Spatial",
    libero_object: "LIBERO-Object", libero_goal: "LIBERO-Goal", libero_10: "LIBERO-Long (10)"
  };
  const DOMAINS = { real_robot: "Real robot", libero: "LIBERO simulation" };
  const state = { videos: [], domain: "real_robot", method: "tcr", task: "all", visible: PAGE_SIZE };
  const $ = (id) => document.getElementById(id);
  const grid = $("recording-grid");
  const methodFilter = $("method-filter");
  const taskFilter = $("task-filter");
  const tabs = [...document.querySelectorAll("[role=tab]")];
  let observer;
  let hasLoaded = false;

  function node(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined) element.textContent = content;
    return element;
  }

  function safeMediaPath(path) {
    if (typeof path !== "string" || !path.startsWith("assets/")) return null;
    const url = new URL(path, document.baseURI);
    const base = new URL(".", document.baseURI);
    return url.origin === base.origin && url.pathname.startsWith(base.pathname + "assets/") ? url.href : null;
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) return "Recording";
    const duration = Math.round(seconds);
    const minutes = Math.floor(duration / 60);
    return `${minutes}:${String(duration % 60).padStart(2, "0")}`;
  }

  function episodeLabel(video) {
    const filename = video.original_filename || video.path.split("/").pop();
    const episode = filename.match(/(?:^|_)ep(\d+)(?:_|\.)/i);
    const task = filename.match(/(?:^|_)task(\d+)(?:_|\.)/i);
    const repeat = filename.match(/(?:^|_)r(\d+)(?:_|\.)/i);
    if (episode) {
      const parts = [];
      if (task) parts.push(`Task ${task[1]}`);
      if (repeat) parts.push(`Run ${repeat[1]}`);
      parts.push(`Episode ${episode[1]}`);
      return parts.join(" · ");
    }
    return `Recording · ${filename.replace(/\.[^.]+$/, "")}`;
  }

  function activateVideo(video) {
    if (video.dataset.src) {
      video.src = video.dataset.src;
      delete video.dataset.src;
    }
  }

  function createCard(recording) {
    const card = node("article", `recording-card${recording.method === "tcr" ? " is-tcr" : ""}`);
    const frame = node("div", "video-frame");
    const video = node("video");
    video.controls = true;
    video.preload = "none";
    video.playsInline = true;
    video.setAttribute("aria-label", `${METHODS[recording.method] || recording.method}, ${TASKS[recording.task] || recording.task}, ${episodeLabel(recording)}`);
    video.dataset.src = safeMediaPath(recording.path);
    if (safeMediaPath(recording.poster)) video.poster = safeMediaPath(recording.poster);
    video.addEventListener("pointerdown", () => activateVideo(video), { once: true });
    video.addEventListener("focus", () => activateVideo(video), { once: true });
    video.addEventListener("play", () => {
      grid.querySelectorAll("video").forEach((other) => { if (other !== video) other.pause(); });
    });
    frame.append(video);

    const content = node("div", "card-content");
    const topline = node("div", "card-topline");
    topline.append(node("span", "method-badge", METHODS[recording.method] || recording.method));
    if (recording.domain === "libero" && ["success", "failure"].includes(recording.outcome)) {
      const outcome = node("span", `outcome ${recording.outcome}`, recording.outcome === "success" ? "Success" : "Failure");
      outcome.title = "Outcome label from the supplied recording filename; not an aggregate score.";
      topline.append(outcome);
    }
    content.append(topline, node("h4", "card-title", TASKS[recording.task] || recording.task), node("p", "episode-title", episodeLabel(recording)));
    const bottomline = node("div", "card-bottomline");
    const duration = node("time", "", formatDuration(recording.duration));
    if (Number.isFinite(recording.duration) && recording.duration > 0) duration.dateTime = `PT${recording.duration}S`;
    const open = node("a", "", "Open video ↗");
    open.href = safeMediaPath(recording.path);
    open.target = "_blank";
    open.rel = "noopener";
    open.setAttribute("aria-label", `Open ${video.getAttribute("aria-label")} in a new tab`);
    bottomline.append(duration, open);
    content.append(bottomline);
    video.addEventListener("error", () => {
      if (!content.querySelector(".video-error")) content.append(node("p", "video-error", "Playback is unavailable in this browser. Use “Open video” to view or download the recording."));
    });
    card.append(frame, content);
    return card;
  }

  function filteredVideos() {
    return state.videos.filter((video) => video.domain === state.domain && (state.task === "all" || video.task === state.task) && (state.method === "all" || video.method === state.method));
  }

  function addCards(videos) {
    const fragment = document.createDocumentFragment();
    videos.forEach((video) => fragment.append(createCard(video)));
    grid.append(fragment);
    grid.querySelectorAll("video[data-src]").forEach((video) => {
      if (observer) observer.observe(video);
      else activateVideo(video);
    });
  }

  function updateCounts(total) {
    const shown = Math.min(state.visible, total);
    $("result-count").textContent = `${total} recording${total === 1 ? "" : "s"}`;
    $("shown-count").textContent = total ? `Showing ${shown} of ${total} recordings` : "";
    $("load-more").hidden = shown >= total;
    $("load-more").textContent = `Show ${Math.min(PAGE_SIZE, total - shown)} more recordings ↓`;
  }

  function updateUrl() {
    const url = new URL(window.location.href);
    url.searchParams.set("domain", state.domain);
    url.searchParams.set("method", state.method);
    if (state.task === "all") url.searchParams.delete("task");
    else url.searchParams.set("task", state.task);
    window.history.replaceState(null, "", url);
  }

  function render() {
    if (!hasLoaded) return;
    state.visible = PAGE_SIZE;
    if (observer) observer.disconnect();
    grid.querySelectorAll("video").forEach((video) => { video.pause(); video.removeAttribute("src"); video.load(); });
    grid.replaceChildren();
    const filtered = filteredVideos();
    $("results-title").textContent = `${state.method === "all" ? "All methods" : METHODS[state.method] || state.method} · ${state.task === "all" ? DOMAINS[state.domain] : TASKS[state.task] || state.task}`;
    $("empty-state").hidden = filtered.length !== 0;
    $("domain-note").textContent = state.domain === "real_robot"
      ? "Browse individual real-robot recordings. Task names follow the supplied collection; no outcome labels are assigned to these clips."
      : "Success / failure labels come from the supplied filenames. These selected episodes are qualitative examples, not aggregate success rates.";
    addCards(filtered.slice(0, PAGE_SIZE));
    updateCounts(filtered.length);
    updateUrl();
  }

  function updateTaskOptions() {
    taskFilter.replaceChildren(new Option(state.domain === "libero" ? "All suites" : "All tasks", "all"));
    Object.entries(TASKS).filter(([task]) => state.videos.some((video) => video.domain === state.domain && video.task === task))
      .forEach(([value, label]) => taskFilter.add(new Option(label, value)));
    if (![...taskFilter.options].some((option) => option.value === state.task)) state.task = "all";
    taskFilter.value = state.task;
  }

  function setDomain(domain) {
    state.domain = domain;
    tabs.forEach((tab) => {
      const selected = tab.dataset.domain === domain;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected) $("recording-panel").setAttribute("aria-labelledby", tab.id);
    });
    updateTaskOptions();
    render();
  }

  async function load() {
    $("error-state").hidden = true;
    $("result-count").textContent = "Loading the collection…";
    grid.setAttribute("aria-busy", "true");
    try {
      const response = await fetch("assets/media.json");
      if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);
      const manifest = await response.json();
      if (manifest.version !== 1 || !Array.isArray(manifest.videos)) throw new Error("Unsupported media manifest");
      state.videos = manifest.videos.filter((video) => DOMAINS[video.domain] && safeMediaPath(video.path));
      const methodOrder = Object.keys(METHODS);
      const taskOrder = Object.keys(TASKS);
      state.videos.sort((a, b) => methodOrder.indexOf(a.method) - methodOrder.indexOf(b.method) || taskOrder.indexOf(a.task) - taskOrder.indexOf(b.task) || a.path.localeCompare(b.path, undefined, { numeric: true }));
      const methods = new Set(state.videos.map((video) => video.method));
      $("total-count").textContent = state.videos.length;
      $("method-count").textContent = String(methods.size).padStart(2, "0");
      $("real-count").textContent = state.videos.filter((video) => video.domain === "real_robot").length;
      $("libero-count").textContent = state.videos.filter((video) => video.domain === "libero").length;
      methodFilter.replaceChildren(new Option("All methods", "all"));
      Object.entries(METHODS).filter(([method]) => methods.has(method)).forEach(([value, label]) => methodFilter.add(new Option(label, value)));
      const params = new URLSearchParams(window.location.search);
      state.domain = DOMAINS[params.get("domain")] ? params.get("domain") : "real_robot";
      state.method = params.get("method") === "all" || methods.has(params.get("method")) ? params.get("method") : "tcr";
      state.task = TASKS[params.get("task")] ? params.get("task") : "all";
      methodFilter.value = state.method;
      methodFilter.disabled = false;
      taskFilter.disabled = false;
      $("reset-filters").disabled = false;
      hasLoaded = true;
      setDomain(state.domain);
    } catch (error) {
      $("error-state").hidden = false;
      $("result-count").textContent = "Collection unavailable";
      $("error-message").textContent = window.location.protocol === "file:"
        ? "Open this gallery through a local HTTP server or GitHub Pages. From the repository folder, run python -m http.server 8000, then open http://localhost:8000/gallery.html."
        : "The recording index could not be loaded. Check your connection and try again, or browse the video links in the README.";
      console.error("TCR gallery:", error);
    } finally {
      grid.setAttribute("aria-busy", "false");
    }
  }

  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver((entries) => {
      entries.filter((entry) => entry.isIntersecting).forEach((entry) => {
        activateVideo(entry.target);
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "240px 0px" });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => setDomain(tab.dataset.domain));
    tab.addEventListener("keydown", (event) => {
      const next = event.key === "ArrowRight" ? (index + 1) % tabs.length
        : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length
          : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
      if (next !== null) { event.preventDefault(); tabs[next].focus(); setDomain(tabs[next].dataset.domain); }
    });
  });
  methodFilter.addEventListener("change", () => { state.method = methodFilter.value; render(); });
  taskFilter.addEventListener("change", () => { state.task = taskFilter.value; render(); });
  $("reset-filters").addEventListener("click", () => { state.method = "tcr"; state.task = "all"; methodFilter.value = "tcr"; taskFilter.value = "all"; render(); });
  $("show-all-methods").addEventListener("click", () => { state.method = "all"; methodFilter.value = "all"; render(); methodFilter.focus(); });
  $("retry-load").addEventListener("click", load);
  $("load-more").addEventListener("click", () => {
    const filtered = filteredVideos();
    const firstNewIndex = grid.children.length;
    addCards(filtered.slice(state.visible, state.visible + PAGE_SIZE));
    state.visible += PAGE_SIZE;
    updateCounts(filtered.length);
    grid.children[firstNewIndex]?.querySelector("video")?.focus({ preventScroll: true });
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) grid.querySelectorAll("video").forEach((video) => video.pause()); });
  load();
})();
