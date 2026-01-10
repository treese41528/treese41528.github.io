document.addEventListener('DOMContentLoaded', function() {
  // ----- Accessibility tweaks -----

  // 1) Search input ARIA
  var searchInput = document.querySelector('.wy-side-nav-search input[type="text"], .bd-search input[type="search"]');
  if (searchInput) {
    searchInput.setAttribute('role', 'searchbox');
    searchInput.setAttribute('aria-label', 'Search');
  }

  // 2) Navigation ARIA labels
  var navElements = document.querySelectorAll('[role="navigation"]');
  Array.prototype.forEach.call(navElements, function(nav) {
    if (!nav.hasAttribute('aria-label')) {
      nav.setAttribute('aria-label', 'Main navigation');
    }
  });

  // 3) Headings missing aria-level
  var headings = document.querySelectorAll('[role="heading"]:not([aria-level])');
  Array.prototype.forEach.call(headings, function(heading) {
    heading.setAttribute('aria-level', '2');
  });

  // ----- Scrollable focusability -----

  // Helper to detect overflow
  function isOverflow(el) {
    var cs = window.getComputedStyle ? getComputedStyle(el) : el.style;
    var overflowX = cs.overflowX || 'visible';
    var overflowY = cs.overflowY || 'visible';
    var overflowVisible = (overflowX === 'visible' && overflowY === 'visible');
    var dimsOverflow = (el.scrollHeight > el.clientHeight) || (el.scrollWidth > el.clientWidth);
    return (!overflowVisible) && dimsOverflow;
  }

  // 1) Scrollable sidebars
  var sidebars = document.querySelectorAll('.wy-side-scroll, .bd-sidebar__content, .sidebar-scroll');
  Array.prototype.forEach.call(sidebars, function(el) {
    if (isOverflow(el) && !el.hasAttribute('tabindex')) {
      el.setAttribute('tabindex', '0');
      el.setAttribute('role', el.getAttribute('role') || 'navigation');
      if (!el.hasAttribute('aria-label')) el.setAttribute('aria-label', 'Section navigation');
    }
  });

  // 2) Overflowing code blocks
  var pres = document.querySelectorAll('div.highlight pre, pre');
  Array.prototype.forEach.call(pres, function(pre) {
    var overflow = pre.scrollWidth > pre.clientWidth || pre.scrollHeight > pre.clientHeight;
    if (overflow && !pre.hasAttribute('tabindex')) {
      pre.setAttribute('tabindex', '0');
      if (!pre.hasAttribute('role')) pre.setAttribute('role', 'region');
      if (!pre.hasAttribute('aria-label')) pre.setAttribute('aria-label', 'Code example');
    }
  });

  // 3) Any inline-styled overflow panels
  var overflowEls = document.querySelectorAll('[style*="overflow"]');
  Array.prototype.forEach.call(overflowEls, function(el) {
    var cs = window.getComputedStyle ? getComputedStyle(el) : el.style;
    var overflow = (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') &&
                   (el.scrollHeight > el.clientHeight || el.scrollWidth > el.clientWidth);
    if (overflow && !el.hasAttribute('tabindex')) {
      el.setAttribute('tabindex', '0');
      if (!el.hasAttribute('role')) el.setAttribute('role', 'region');
      if (!el.hasAttribute('aria-label')) el.setAttribute('aria-label', 'Scrollable content');
    }
  });

  // ----- Iframe titles (generic, not course-specific) -----

  // Optional global mapping you can define elsewhere:
  // window.VIDEO_TITLE_MAP = { "<id>": "Your human-friendly title", ... }
  var videoTitleMap = (window.VIDEO_TITLE_MAP || {});
  var fallbackTitles = {
    youtube: 'Embedded YouTube video',
    vimeo: 'Embedded Vimeo video',
    generic: 'Embedded media'
  };

  function platformFromSrc(src) {
    if (!src) return 'generic';
    if (src.indexOf('youtube') !== -1 || src.indexOf('youtu.be') !== -1) return 'youtube';
    if (src.indexOf('vimeo') !== -1) return 'vimeo';
    return 'generic';
  }

  function parseYouTubeId(src) {
    if (!src) return null;
    var m;
    m = src.match(/embed\/([^?/#]+)/);
    if (m && m[1]) return m[1];
    m = src.match(/watch\?v=([^&?#]+)/);
    if (m && m[1]) return m[1];
    m = src.match(/youtu\.be\/([^?/#]+)/);
    if (m && m[1]) return m[1];
    return null;
  }

  function parseVimeoId(src) {
    if (!src) return null;
    var m = src.match(/vimeo\.com\/(?:video\/)?(\d+)/);
    return (m && m[1]) ? m[1] : null;
    }

  // Add titles to iframes lacking a title
  var iframes = document.querySelectorAll('iframe:not([title])');
  Array.prototype.forEach.call(iframes, function(iframe) {
    // 0) Allow explicit per-iframe override
    var dataTitle = iframe.getAttribute('data-title');
    if (dataTitle) {
      iframe.setAttribute('title', dataTitle);
      return;
    }

    var src = iframe.getAttribute('src') || '';
    var platform = platformFromSrc(src);

    if (platform === 'youtube') {
      var yid = parseYouTubeId(src);
      if (yid && videoTitleMap[yid]) {
        iframe.setAttribute('title', videoTitleMap[yid]);
      } else {
        iframe.setAttribute('title', fallbackTitles.youtube);
      }
      return;
    }

    if (platform === 'vimeo') {
      var vid = parseVimeoId(src);
      if (vid && videoTitleMap[vid]) {
        iframe.setAttribute('title', videoTitleMap[vid]);
      } else {
        iframe.setAttribute('title', fallbackTitles.vimeo);
      }
      return;
    }

    iframe.setAttribute('title', fallbackTitles.generic);
  });
});
