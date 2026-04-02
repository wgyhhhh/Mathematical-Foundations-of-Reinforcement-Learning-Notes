window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
    enableMenu: false,
    renderActions: {
      assistiveMml: []
    }
  },
  startup: {
    typeset: false
  }
};

let mathJaxRenderQueue = Promise.resolve();

function getMathElements(root) {
  if (!root || !root.querySelectorAll) {
    return [];
  }

  return Array.from(root.querySelectorAll(".arithmatex")).filter(
    (element) => !element.hasAttribute("data-mathjax-processed")
  );
}

function renderMath(root) {
  if (!window.MathJax || !window.MathJax.typesetPromise) {
    return;
  }

  const elements = getMathElements(root);
  if (!elements.length) {
    return;
  }

  mathJaxRenderQueue = mathJaxRenderQueue
    .then(() => window.MathJax.typesetPromise(elements))
    .then(() => {
      elements.forEach((element) => {
        element.setAttribute("data-mathjax-processed", "true");
      });
    })
    .catch((error) => {
      console.error("MathJax render failed:", error);
      elements.forEach((element) => {
        element.removeAttribute("data-mathjax-processed");
      });
    });
}

document$.subscribe((document) => {
  renderMath(document);
});
