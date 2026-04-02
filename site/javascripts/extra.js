// https://github.com/ColdDay/click-colorful

(function (win, doc) {
  "use strict";
  var defaultParams = {
    colors: ['#eb125f', '#6eff8a', '#6386ff', '#f9f383'],
    size: 10,
    maxCount: 24
  }
  function colorBall(params) {
    this.params = Object.assign({}, defaultParams, params)
  }
  function getOneRandom(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }
  function _run(ball) {
    var randomXFlag = Math.random() > 0.5
    var randomYFlag = Math.random() > 0.5
    var randomX = parseInt(Math.random() * 160);
    var randomY = parseInt(Math.random() * 160);
    if (randomXFlag) {
      randomX = randomX * -1;
    }
    if (randomYFlag) {
      randomY = randomY * -1
    }
    var transform = 'translate3d(' + randomX + 'px,' + randomY + 'px, 0) scale(0)';
    ball.style.webkitTransform = transform;
    ball.style.MozTransform = transform;
    ball.style.msTransform = transform;
    ball.style.OTransform = transform;
    ball.style.transform = transform;
  }
  colorBall.prototype.fly = function (x, y, playCount, loopTimer) {
    if (!loopTimer) loopTimer = 300
    var ballElements = []
    var fragment = document.createDocumentFragment()

    var ballNum = this.params.maxCount;
    // 修改轮换播放实现方式，改为一次创建所有，通过延迟执行动画实现
    if (playCount) {
      ballNum = ballNum * playCount;
    }
    var loop = 0
    for (var i = 0; i < ballNum; i++) {
      var curLoop = parseInt(i / this.params.maxCount)
      var ball = doc.createElement('i');
      ball.className = 'color-ball ball-loop-' + curLoop;
      var blurX = Math.random() * 10
      if (Math.random() > 0.5) blurX = blurX * -1
      var blurY = Math.random() * 10
      if (Math.random() > 0.5) blurY = blurY * -1
      ball.style.left = (x) + 'px';
      ball.style.top = (y) + 'px';
      ball.style.width = this.params.size + 'px';
      ball.style.height = this.params.size + 'px';
      ball.style.position = 'fixed';
      ball.style.borderRadius = '1000px';
      ball.style.boxSizing = 'border-box';
      ball.style.zIndex = 9999;
      ball.style.opacity = 0;
      if (curLoop === 0) ball.style.opacity = 1;
      ball.style.transform = 'translate3d(0px, 0px, 0px) scale(1)';
      ball.style.webkitTransform = 'translate3d(0px, 0px, 0px) scale(1)';
      ball.style.transition = 'transform 1s ' + curLoop * loopTimer / 1000 + 's ease-out';
      ball.style.webkitTransition = 'transform 1s ' + curLoop * loopTimer / 1000 + 's ease-out';
      ball.style.backgroundColor = getOneRandom(this.params.colors);
      fragment.appendChild(ball);
      ballElements.push(ball)
      // 性能优化终极版
      if (curLoop !== loop) {
        (function (num) {
          setTimeout(function () {
            var loopBalls = document.getElementsByClassName('ball-loop-' + num)
            for (var j = 0; j < loopBalls.length; j++) {
              loopBalls[j].style.opacity = 1
            }
            if (num === loop) {
              _clear(ballElements)
            }
          }, num * loopTimer + 30)
        })(curLoop)
        loop = curLoop
      }
    }

    doc.body.appendChild(fragment);
    // 延迟删除
    !playCount && _clear(ballElements)
    // 执行动画
    setTimeout(function () {
      for (var i = 0; i < ballElements.length; i++) {
        _run(ballElements[i])
      }
    }, 10)
  }
  function _clear(balls) {
    setTimeout(function () {
      for (var i = 0; i < balls.length; i++) {
        doc.body.removeChild(balls[i])
      }
    }, 1000)

  }
  //兼容CommonJs规范 
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = colorBall;
  };
  //兼容AMD/CMD规范
  if (typeof define === 'function') define(function () {
    return colorBall;
  });
  //注册全局变量，兼容直接使用script标签引入插件
  win.colorBall = colorBall;
})(window, document)

window.addEventListener(
  "mouseup",
  function (e) {
    var color = new colorBall();
    color.fly(e.clientX, e.clientY);
  },
)

// Inject a searchable input into the top tabs bar and use MkDocs search index.
(function () {
  var searchDocs = [];
  var searchLoaded = false;
  var activeRoot = null;
  var activeInput = null;
  var activeList = null;
  var debounceTimer = null;

  function getBaseUrl() {
    var base = window.__TOPBAR_BASE_URL__ || ".";
    return base.replace(/\/$/, "");
  }

  function toHref(location) {
    if (!location) return "#";
    if (/^(https?:)?\/\//.test(location) || location.charAt(0) === "/") return location;
    var base = getBaseUrl();
    return (base ? base + "/" : "") + location.replace(/^\.\//, "");
  }

  function htmlEscape(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function scoreDoc(doc, query) {
    var title = (doc.title || "").toLowerCase();
    var text = (doc.text || "").toLowerCase();
    var score = 0;

    if (title.indexOf(query) !== -1) score += 20;
    if (title.indexOf(query) === 0) score += 20;
    if (text.indexOf(query) !== -1) score += 8;
    return score;
  }

  function loadSearchIndex() {
    if (searchLoaded) return Promise.resolve();
    var indexUrl = window.__TOPBAR_SEARCH_INDEX_URL__;
    if (!indexUrl) return Promise.resolve();

    return fetch(indexUrl)
      .then(function (res) {
        if (!res.ok) throw new Error("Failed to load search index");
        return res.json();
      })
      .then(function (json) {
        searchDocs = Array.isArray(json.docs) ? json.docs : [];
        searchLoaded = true;
      })
      .catch(function () {
        searchDocs = [];
        searchLoaded = false;
      });
  }

  function hideResults() {
    if (activeRoot) activeRoot.classList.remove("is-open");
    if (activeList) activeList.innerHTML = "";
  }

  function renderResults(query) {
    if (!activeList) return;
    var q = query.trim().toLowerCase();
    if (!q) {
      hideResults();
      return;
    }

    var matches = searchDocs
      .map(function (doc) {
        return { doc: doc, score: scoreDoc(doc, q) };
      })
      .filter(function (item) {
        return item.score > 0;
      })
      .sort(function (a, b) {
        return b.score - a.score;
      })
      .slice(0, 8);

    if (!matches.length) {
      activeList.innerHTML = '<li class="topbar-search__empty">未找到相关内容</li>';
      activeRoot.classList.add("is-open");
      return;
    }

    activeList.innerHTML = matches
      .map(function (item) {
        var title = htmlEscape(item.doc.title || "未命名页面");
        var text = htmlEscape((item.doc.text || "").replace(/\s+/g, " ").trim().slice(0, 84));
        var href = htmlEscape(toHref(item.doc.location));
        return (
          '<li class="topbar-search__item">' +
          '<a class="topbar-search__link" href="' +
          href +
          '">' +
          '<span class="topbar-search__title">' +
          title +
          "</span>" +
          '<span class="topbar-search__snippet">' +
          text +
          "</span>" +
          "</a>" +
          "</li>"
        );
      })
      .join("");

    activeRoot.classList.add("is-open");
  }

  function bindEvents() {
    if (!activeInput) return;

    activeInput.addEventListener("input", function () {
      var value = activeInput.value || "";
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function () {
        renderResults(value);
      }, 100);
    });

    activeInput.addEventListener("focus", function () {
      renderResults(activeInput.value || "");
    });

    document.addEventListener("click", function (e) {
      if (!activeRoot || !activeRoot.contains(e.target)) hideResults();
    });

    activeInput.addEventListener("keydown", function (e) {
      if (e.key === "Escape") hideResults();
    });
  }

  function mountSearchBar() {
    var tabsInner = document.querySelector(".md-tabs__inner");
    if (!tabsInner) return;
    if (tabsInner.querySelector(".topbar-search")) return;

    var root = document.createElement("div");
    root.className = "topbar-search";
    root.innerHTML =
      '<input class="topbar-search__input" type="search" placeholder="搜索章节内容..." aria-label="搜索章节内容" />' +
      '<ul class="topbar-search__results" aria-label="搜索结果"></ul>';

    tabsInner.appendChild(root);

    activeRoot = root;
    activeInput = root.querySelector(".topbar-search__input");
    activeList = root.querySelector(".topbar-search__results");

    loadSearchIndex().finally(bindEvents);
  }

  function init() {
    mountSearchBar();
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(init);
  } else {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", init);
    } else {
      init();
    }
  }
})();
