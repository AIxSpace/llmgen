// 代码高亮(Zig 用自定义的简易语法) + 复制按钮
(function () {
  window.addEventListener('DOMContentLoaded', function () {
    if (window.hljs) {
      hljs.registerLanguage('zig', function (h) {
        return {
          name: 'Zig',
          keywords: {
            keyword: 'const var fn pub return if else while for switch try catch orelse defer errdefer struct enum union error test comptime inline extern export break continue unreachable and or',
            type: 'f32 f64 f16 u8 u16 u32 u64 i8 i16 i32 i64 usize isize bool void anyopaque anyerror type c_int noreturn',
            literal: 'true false null undefined'
          },
          contains: [
            h.COMMENT('//', '$'),
            h.QUOTE_STRING_MODE,
            h.C_NUMBER_MODE,
            { className: 'built_in', begin: /@[A-Za-z_]+/ },
            { className: 'title', begin: /\b[a-zA-Z_]\w*(?=\()/ }
          ]
        };
      });
      document.querySelectorAll('pre code').forEach(function (el) { hljs.highlightElement(el); });
    }
    document.querySelectorAll('.codeblock').forEach(function (blk) {
      var btn = blk.querySelector('.cap button');
      var code = blk.querySelector('pre code');
      if (!btn || !code) return;
      btn.addEventListener('click', function () {
        var text = code.innerText;
        var done = function () { btn.textContent = '已复制'; setTimeout(function () { btn.textContent = '复制'; }, 1200); };
        if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
      });
    });
  });
})();
