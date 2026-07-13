// Minimal, dependency-free. Only job: respect prefers-reduced-motion for the
// autoplaying hero video (matches the same convention used on
// hksaperstein.github.io's FeaturedProject component).
(function () {
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.querySelectorAll('video[autoplay]').forEach(function (video) {
      video.removeAttribute('autoplay');
      video.pause();
      video.setAttribute('controls', '');
    });
  }
})();
