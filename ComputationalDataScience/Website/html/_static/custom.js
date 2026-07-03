document.addEventListener('DOMContentLoaded', function() {
  // ----- Accessibility tweaks -----

  // 0) Skip-to-main-content link (WCAG 2.4.1 Bypass Blocks). The RTD theme renders
  // the full multi-level toctree before the content, so keyboard users need a way past it.
  var mainRegion = document.querySelector('[role="main"]');
  if (mainRegion) {
    if (!mainRegion.id) mainRegion.id = 'main-content';
    var skip = document.createElement('a');
    skip.className = 'skip-link';
    skip.href = '#' + mainRegion.id;
    skip.textContent = 'Skip to main content';
    document.body.insertBefore(skip, document.body.firstChild);
  }

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

  // 2) Overflowing code blocks — number the labels so each scrollable region
  // landmark has a UNIQUE accessible name (axe "landmark-unique").
  var pres = document.querySelectorAll('div.highlight pre, pre');
  var codeIdx = 0;
  Array.prototype.forEach.call(pres, function(pre) {
    var overflow = pre.scrollWidth > pre.clientWidth || pre.scrollHeight > pre.clientHeight;
    if (overflow && !pre.hasAttribute('tabindex')) {
      codeIdx++;
      pre.setAttribute('tabindex', '0');
      if (!pre.hasAttribute('role')) pre.setAttribute('role', 'region');
      if (!pre.hasAttribute('aria-label')) pre.setAttribute('aria-label', 'Code example ' + codeIdx);
    }
  });

  // 3) Any inline-styled overflow panels — likewise uniquely labelled.
  var overflowEls = document.querySelectorAll('[style*="overflow"]');
  var scrollIdx = 0;
  Array.prototype.forEach.call(overflowEls, function(el) {
    var cs = window.getComputedStyle ? getComputedStyle(el) : el.style;
    var overflow = (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') &&
                   (el.scrollHeight > el.clientHeight || el.scrollWidth > el.clientWidth);
    if (overflow && !el.hasAttribute('tabindex')) {
      scrollIdx++;
      el.setAttribute('tabindex', '0');
      if (!el.hasAttribute('role')) el.setAttribute('role', 'region');
      if (!el.hasAttribute('aria-label')) el.setAttribute('aria-label', 'Scrollable content ' + scrollIdx);
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

  // ----- Exercise and Solution Controls (sphinx-design dropdowns) -----

  // Find all exercise admonitions
  var exercises = document.querySelectorAll('div.admonition.exercise');
  
  // Only add controls if there are 3+ exercises on the page
  if (exercises.length >= 3) {
    var firstExercise = exercises[0];
    
    // Create control buttons container
    var controlDiv = document.createElement('div');
    controlDiv.className = 'exercise-controls';
    controlDiv.setAttribute('role', 'group');
    controlDiv.setAttribute('aria-label', 'Solution visibility controls');
    
    var expandBtn = document.createElement('button');
    expandBtn.type = 'button';
    expandBtn.className = 'expand-all-solutions';
    expandBtn.textContent = 'Show All Solutions';
    
    var collapseBtn = document.createElement('button');
    collapseBtn.type = 'button';
    collapseBtn.className = 'collapse-all-solutions';
    collapseBtn.textContent = 'Hide All Solutions';
    
    controlDiv.appendChild(expandBtn);
    controlDiv.appendChild(document.createTextNode(' '));
    controlDiv.appendChild(collapseBtn);
    
    // Insert before first exercise
    firstExercise.parentNode.insertBefore(controlDiv, firstExercise);
    
    // Expand all solutions
    expandBtn.addEventListener('click', function() {
      // sphinx-design dropdowns use <details> elements with class sd-dropdown
      var dropdowns = document.querySelectorAll('div.admonition.exercise details.sd-dropdown');
      Array.prototype.forEach.call(dropdowns, function(details) {
        details.open = true;
      });
      
      // Also try more general selector for any details in exercises
      var allDetails = document.querySelectorAll('div.admonition.exercise details');
      Array.prototype.forEach.call(allDetails, function(details) {
        details.open = true;
      });
      
      // Also handle any sphinx-togglebutton elements if present
      var toggleHidden = document.querySelectorAll('div.admonition.exercise .toggle.toggle-hidden');
      Array.prototype.forEach.call(toggleHidden, function(el) {
        el.classList.remove('toggle-hidden');
      });
      
      // Update all summary aria-expanded attributes
      var summaries = document.querySelectorAll('div.admonition.exercise details summary');
      Array.prototype.forEach.call(summaries, function(summary) {
        summary.setAttribute('aria-expanded', 'true');
      });
    });
    
    // Collapse all solutions
    collapseBtn.addEventListener('click', function() {
      // sphinx-design dropdowns
      var dropdowns = document.querySelectorAll('div.admonition.exercise details.sd-dropdown');
      Array.prototype.forEach.call(dropdowns, function(details) {
        details.open = false;
      });
      
      // Also try more general selector
      var allDetails = document.querySelectorAll('div.admonition.exercise details');
      Array.prototype.forEach.call(allDetails, function(details) {
        details.open = false;
      });
      
      // Also handle sphinx-togglebutton elements
      var toggleVisible = document.querySelectorAll('div.admonition.exercise .toggle:not(.toggle-hidden)');
      Array.prototype.forEach.call(toggleVisible, function(el) {
        el.classList.add('toggle-hidden');
      });
      
      // Update all summary aria-expanded attributes
      var summaries = document.querySelectorAll('div.admonition.exercise details summary');
      Array.prototype.forEach.call(summaries, function(summary) {
        summary.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // Add ARIA attributes to sphinx-design dropdown summaries for accessibility
  var sdDropdowns = document.querySelectorAll('details.sd-dropdown, details');
  Array.prototype.forEach.call(sdDropdowns, function(details) {
    var summary = details.querySelector('summary');
    if (summary) {
      // Set initial state
      summary.setAttribute('aria-expanded', details.open ? 'true' : 'false');

      // Update on toggle
      details.addEventListener('toggle', function() {
        summary.setAttribute('aria-expanded', details.open ? 'true' : 'false');
      });
    }
  });

  // ----- Landmark regions (WCAG 1.3.1 / 2.4.1) -----

  // The RTD theme leaves the footer's "Built with Sphinx …" text as a direct child
  // of <footer>, outside any landmark (axe "region"). Promote the whole footer to a
  // single contentinfo landmark so all of its content is contained, and demote the
  // theme's inner contentinfo div to avoid duplicate landmarks.
  var footer = document.querySelector('.wy-nav-content footer');
  if (footer) {
    var innerCI = footer.querySelector('[role="contentinfo"]');
    if (innerCI) innerCI.removeAttribute('role');
    footer.setAttribute('role', 'contentinfo');
    // The prev/next buttons are pagination; the theme mislabels them "Footer".
    var pager = footer.querySelector('.rst-footer-buttons[role="navigation"]');
    if (pager) pager.setAttribute('aria-label', 'Pagination');
  }

  // Give every navigation landmark a UNIQUE accessible name (axe "landmark-unique").
  // The sidebar <nav> and any in-page "Contents" <nav> are both unlabelled by default.
  var sideNav = document.querySelector('nav.wy-nav-side');
  if (sideNav && !sideNav.hasAttribute('aria-label')) {
    sideNav.setAttribute('aria-label', 'Documentation sidebar');
  }
  Array.prototype.forEach.call(document.querySelectorAll('nav.contents.local'), function(toc) {
    if (!toc.hasAttribute('aria-label') && !toc.hasAttribute('aria-labelledby')) {
      var title = toc.querySelector('.topic-title');
      toc.setAttribute('aria-label', (title && title.textContent.trim()) || 'Page contents');
    }
  });

  // ----- Bibliography citation roles (WCAG 4.1.2) -----

  // docutils renders citation lists with role="list" whose children carry the
  // DEPRECATED and disallowed roles doc-biblioentry / doc-backlink, which trip axe
  // (aria-required-children, aria-allowed-role, aria-deprecated-role). Strip the
  // ARIA roles — the citations still read correctly as text/links to assistive tech.
  Array.prototype.forEach.call(
    document.querySelectorAll('.citation-list[role="list"], [role="doc-biblioentry"], [role="doc-backlink"]'),
    function(el) { el.removeAttribute('role'); }
  );

  // ----- Force-download notebook .ipynb links -----
  // GitHub Pages serves .ipynb with Content-Type: text/html, so browsers render the
  // file inline instead of downloading it (and the download attribute is bypassed on a
  // direct navigation). Intercept clicks and save the file as a blob so the Download
  // buttons always download. CORS is open (access-control-allow-origin: *) so fetch works.
  Array.prototype.forEach.call(document.querySelectorAll('a[href$=".ipynb"]'), function(a) {
    a.addEventListener('click', function(e) {
      // Respect modified clicks (open-in-new-tab, etc.) and already-handled events.
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      e.preventDefault();
      var url = a.href;
      var name = a.getAttribute('download') || url.split('/').pop().split('?')[0];
      fetch(url).then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.blob();
      }).then(function(blob) {
        var objUrl = URL.createObjectURL(blob);
        var tmp = document.createElement('a');
        tmp.href = objUrl;
        tmp.download = name;
        document.body.appendChild(tmp);
        tmp.click();
        tmp.remove();
        setTimeout(function() { URL.revokeObjectURL(objUrl); }, 1500);
      }).catch(function() { window.open(url, '_blank'); }); // fallback: open in a new tab
    });
  });
});