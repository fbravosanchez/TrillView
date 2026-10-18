
(function () {
  var attached = {};
  
  function getPlotDiv(idx) {
    var els = document.querySelectorAll('[id]');
    for (var i = 0; i < els.length; i++) {
      var raw = els[i].getAttribute('id');
      if (!raw || raw.indexOf('spec-graph') === -1) continue;
      try {
        var obj = JSON.parse(raw);
        if (obj && obj.type === 'spec-graph' && obj.index === idx)
          return els[i].querySelector('.js-plotly-plot') || null;
      } catch (e) {}
    }
    return null;
  }
  
  function setPlayhead(idx, xVal) {
    var pd = getPlotDiv(idx);
    if (!pd || !pd._fullLayout) return;
    if (xVal === null) {
      if (pd._fullLayout.shapes && pd._fullLayout.shapes.length > 0)
        Plotly.relayout(pd, { shapes: [] });
    } else {
      Plotly.relayout(pd, { shapes: [{
        type:'line', xref:'x', yref:'paper',
        x0:xVal, x1:xVal, y0:0, y1:1,
        line:{color:'#e03030', width:2.5}
      }]});
    }
  }
  
  function attachAudio(audio) {
    var idx = parseInt(audio.getAttribute('data-idx'), 10);
    if (attached[idx]) return;
    attached[idx] = true;
    audio.addEventListener('timeupdate', function () {
      if (audio.paused) return;
      var clip = JSON.parse(audio.getAttribute('data-clip') || '{}');
      setPlayhead(idx, (clip.start || 0) + audio.currentTime);
    });
    audio.addEventListener('pause',  function () { setPlayhead(idx, null); });
    audio.addEventListener('ended',  function () { setPlayhead(idx, null); });
    audio.addEventListener('seeked', function () {
      if (audio.paused) setPlayhead(idx, null);
    });
  }
  
  function scanAudio() {
    document.querySelectorAll('audio[data-idx]').forEach(attachAudio);
  }
  
  new MutationObserver(function (mutations) {
    mutations.forEach(function (m) {
      m.removedNodes.forEach(function (n) {
        if (n.querySelectorAll)
          n.querySelectorAll('audio[data-idx]').forEach(function (a) {
            delete attached[parseInt(a.getAttribute('data-idx'), 10)];
          });
      });
    });
    scanAudio();
  }).observe(document.body, { childList: true, subtree: true });
  scanAudio();
  
  document.addEventListener('keydown', function (e) {
    var tag = (e.target || {}).tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === 'PageUp' || e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      var b = document.getElementById('page-prev');
      if (b && !b.disabled) { b.click(); e.preventDefault(); }
    } else if (e.key === 'PageDown' || e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      var b = document.getElementById('page-next');
      if (b && !b.disabled) { b.click(); e.preventDefault(); }
    }
  });
})();
