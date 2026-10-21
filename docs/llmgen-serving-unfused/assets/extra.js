// diff 代码块只复制改后的代码; "你的实测"表格存到 localStorage 并算提升倍数
(function () {
  function num(s) { var m = String(s).replace(/,/g, '').match(/-?\d+(\.\d+)?/); return m ? parseFloat(m[0]) : NaN; }
  window.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.codeblock.diff').forEach(function (blk) {
      var btn = blk.querySelector('.cap button');
      btn.addEventListener('click', function (ev) {
        ev.stopImmediatePropagation();
        var text = Array.prototype.filter.call(blk.querySelectorAll('.ln'), function (l) { return !l.classList.contains('del'); })
          .map(function (l) { return l.textContent; }).join('\n');
        var done = function () { btn.textContent = '已复制'; setTimeout(function () { btn.textContent = '复制改后代码'; }, 1200); };
        if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
      }, true);
    });
    document.querySelectorAll('table.mine tbody tr').forEach(function (tr) {
      var cells = tr.querySelectorAll('td.fill'), gain = tr.querySelector('td.gain');
      function update() {
        if (!gain || cells.length < 2) return;
        var a = num(cells[0].textContent), b = num(cells[1].textContent);
        gain.textContent = (a > 0 && b > 0) ? (b >= a ? (b / a).toFixed(2) + '×' : '÷' + (a / b).toFixed(2)) : '';
      }
      cells.forEach(function (td) {
        try { var v = localStorage.getItem('sv:' + td.dataset.key); if (v) td.textContent = v; } catch (e) {}
        td.addEventListener('input', function () {
          try { localStorage.setItem('sv:' + td.dataset.key, td.textContent.trim()); } catch (e) {}
          update();
        });
      });
      update();
    });
  });
})();
