import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { webcrypto } from "node:crypto";

const require = createRequire(import.meta.url);
const esbuild = require("esbuild");
const postcss = require("postcss");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "../..");
const SRC_APP_PATH = path.join(ROOT_DIR, "src", "App.tsx");
const STYLES_CSS_PATH = path.join(ROOT_DIR, "src", "styles.css");

// ---------------------------------------------------------------------------
// 1. PostCSS AST Loader & Style Query Engine (for #66, #70, #87)
// ---------------------------------------------------------------------------
function loadStylesAst() {
  const cssText = fs.readFileSync(STYLES_CSS_PATH, "utf8");
  const root = postcss.parse(cssText, { from: STYLES_CSS_PATH });

  const topLevelRules = [];
  const mediaBlocks = [];

  root.walkRules((rule) => {
    const decls = {};
    rule.walkDecls((d) => {
      decls[d.prop.toLowerCase()] = d.value;
    });
    const parent = rule.parent;
    if (parent && parent.type === "atrule" && parent.name === "media") {
      mediaBlocks.push({
        params: parent.params,
        selector: rule.selector,
        decls,
      });
    } else {
      topLevelRules.push({
        selector: rule.selector,
        decls,
      });
    }
  });

  function findRules(selectorExact, mediaQuery = null) {
    const src = mediaQuery
      ? mediaBlocks.filter((m) => m.params.replace(/\s+/g, " ").includes(mediaQuery))
      : topLevelRules;
    return src.filter((r) =>
      r.selector
        .split(",")
        .map((s) => s.trim())
        .includes(selectorExact),
    );
  }

  function getMergedDecls(selectorExact, mediaQuery = null) {
    const out = {};
    for (const r of findRules(selectorExact, mediaQuery)) {
      Object.assign(out, r.decls);
    }
    return out;
  }

  function resolveOverlayZIndex(el) {
    // Evaluate top-level CSS rules matching el (e.g. .overlay, .overlay.overlay-topmost)
    let z = "";
    for (const r of topLevelRules) {
      if (!r.decls["z-index"]) continue;
      const sels = r.selector.split(",").map((s) => s.trim());
      for (const s of sels) {
        if (s === ".overlay" && el.classList.contains("overlay")) {
          z = r.decls["z-index"];
        } else if (
          s === ".overlay.overlay-topmost" &&
          el.classList.contains("overlay") &&
          el.classList.contains("overlay-topmost")
        ) {
          z = r.decls["z-index"];
        } else if (s === ".toasts" && el.classList.contains("toasts")) {
          z = r.decls["z-index"];
        }
      }
    }
    return z;
  }

  return {
    root,
    topLevelRules,
    mediaBlocks,
    findRules,
    getMergedDecls,
    resolveOverlayZIndex,
  };
}

const STYLES_AST = loadStylesAst();

// ---------------------------------------------------------------------------
// 2. Self-Contained MiniDOM + Event/Focus/Timer/Storage Harness
// ---------------------------------------------------------------------------
function splitTopLevelCommas(selector) {
  const parts = [];
  let cur = "";
  let depthParen = 0;
  let depthBracket = 0;
  let quote = null;
  for (let i = 0; i < selector.length; i++) {
    const ch = selector[i];
    if (quote) {
      cur += ch;
      if (ch === quote && selector[i - 1] !== "\\") quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      cur += ch;
    } else if (ch === "(") {
      depthParen++;
      cur += ch;
    } else if (ch === ")") {
      depthParen = Math.max(0, depthParen - 1);
      cur += ch;
    } else if (ch === "[") {
      depthBracket++;
      cur += ch;
    } else if (ch === "]") {
      depthBracket = Math.max(0, depthBracket - 1);
      cur += ch;
    } else if (ch === "," && depthParen === 0 && depthBracket === 0) {
      if (cur.trim()) parts.push(cur.trim());
      cur = "";
    } else {
      cur += ch;
    }
  }
  if (cur.trim()) parts.push(cur.trim());
  return parts;
}

function tokenizeComplexSelector(complexSel) {
  const tokens = [];
  let cur = "";
  let depthParen = 0;
  let depthBracket = 0;
  let quote = null;
  const s = complexSel.trim();
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quote) {
      cur += ch;
      if (ch === quote && s[i - 1] !== "\\") quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      cur += ch;
    } else if (ch === "(") {
      depthParen++;
      cur += ch;
    } else if (ch === ")") {
      depthParen = Math.max(0, depthParen - 1);
      cur += ch;
    } else if (ch === "[") {
      depthBracket++;
      cur += ch;
    } else if (ch === "]") {
      depthBracket = Math.max(0, depthBracket - 1);
      cur += ch;
    } else if (depthParen === 0 && depthBracket === 0 && (ch === ">" || /\s/.test(ch))) {
      if (cur) {
        tokens.push({ type: "compound", value: cur });
        cur = "";
      }
      if (ch === ">") {
        tokens.push({ type: "combinator", value: ">" });
      } else {
        const prev = tokens[tokens.length - 1];
        if (!prev || prev.type !== "combinator") {
          tokens.push({ type: "combinator", value: " " });
        }
      }
    } else {
      cur += ch;
    }
  }
  if (cur) tokens.push({ type: "compound", value: cur });
  // Collapse whitespace around '>'
  const collapsed = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type === "combinator" && t.value === " ") {
      const prev = collapsed[collapsed.length - 1];
      const next = tokens[i + 1];
      if ((prev && prev.type === "combinator") || (next && next.type === "combinator")) {
        continue;
      }
    }
    collapsed.push(t);
  }
  return collapsed;
}

function matchCompound(el, compound, scopeNode) {
  if (!el || el.nodeType !== 1) return false;
  let i = 0;
  const s = compound;
  while (i < s.length) {
    const ch = s[i];
    if (ch === "*") {
      i++;
    } else if (ch === "#") {
      i++;
      const start = i;
      while (i < s.length && !/[.#[:\s>]/.test(s[i])) i++;
      const id = s.slice(start, i);
      if (el.id !== id) return false;
    } else if (ch === ".") {
      i++;
      const start = i;
      while (i < s.length && !/[.#[:\s>]/.test(s[i])) i++;
      const cls = s.slice(start, i);
      if (!el.classList.contains(cls)) return false;
    } else if (ch === "[") {
      let end = i + 1;
      let q = null;
      while (end < s.length) {
        if (q) {
          if (s[end] === q && s[end - 1] !== "\\") q = null;
        } else if (s[end] === '"' || s[end] === "'") {
          q = s[end];
        } else if (s[end] === "]") {
          break;
        }
        end++;
      }
      const inside = s.slice(i + 1, end).trim();
      i = end + 1;
      const eqIdx = inside.indexOf("=");
      if (eqIdx === -1) {
        if (!el.hasAttribute(inside)) return false;
      } else {
        let op = "=";
        let attrName = inside.slice(0, eqIdx).trim();
        if (attrName.endsWith("*") || attrName.endsWith("^") || attrName.endsWith("$")) {
          op = attrName.slice(-1) + "=";
          attrName = attrName.slice(0, -1).trim();
        }
        let attrVal = inside.slice(eqIdx + 1).trim();
        if (
          (attrVal.startsWith('"') && attrVal.endsWith('"')) ||
          (attrVal.startsWith("'") && attrVal.endsWith("'"))
        ) {
          attrVal = attrVal.slice(1, -1);
        }
        const actual = el.getAttribute(attrName);
        if (actual === null) return false;
        if (op === "=" && actual !== attrVal) return false;
        if (op === "*=" && !actual.includes(attrVal)) return false;
        if (op === "^=" && !actual.startsWith(attrVal)) return false;
        if (op === "$=" && !actual.endsWith(attrVal)) return false;
      }
    } else if (ch === ":") {
      i++;
      const start = i;
      while (i < s.length && !/[.#[:(\s>]/.test(s[i])) i++;
      const pseudo = s.slice(start, i);
      let arg = null;
      if (s[i] === "(") {
        let depth = 1;
        let p = i + 1;
        let q = null;
        while (p < s.length && depth > 0) {
          if (q) {
            if (s[p] === q && s[p - 1] !== "\\") q = null;
          } else if (s[p] === '"' || s[p] === "'") {
            q = s[p];
          } else if (s[p] === "(") depth++;
          else if (s[p] === ")") depth--;
          p++;
        }
        arg = s.slice(i + 1, p - 1);
        i = p;
      }
      if (pseudo === "disabled") {
        if (!el.hasAttribute("disabled") && !el.disabled) return false;
      } else if (pseudo === "enabled") {
        if (el.hasAttribute("disabled") || el.disabled) return false;
      } else if (pseudo === "scope") {
        if (el !== scopeNode) return false;
      } else if (pseudo === "not") {
        if (arg && matchSelector(el, arg, scopeNode)) return false;
      } else if (pseudo === "checked") {
        if (!el.checked) return false;
      }
    } else {
      const start = i;
      while (i < s.length && !/[.#[:\s>]/.test(s[i])) i++;
      const tag = s.slice(start, i).toLowerCase();
      if (el.tagName.toLowerCase() !== tag) return false;
    }
  }
  return true;
}

function matchComplex(el, tokens, scopeNode) {
  if (!el || el.nodeType !== 1 || tokens.length === 0) return false;
  function matchAt(node, idx) {
    const tok = tokens[idx];
    if (!tok || tok.type !== "compound") return false;
    if (!matchCompound(node, tok.value, scopeNode)) return false;
    if (idx === 0) return true;
    const comb = tokens[idx - 1];
    if (comb.value === ">") {
      return node.parentNode && node.parentNode.nodeType === 1
        ? matchAt(node.parentNode, idx - 2)
        : false;
    }
    let anc = node.parentNode;
    while (anc && anc.nodeType === 1) {
      if (matchAt(anc, idx - 2)) return true;
      anc = anc.parentNode;
    }
    return false;
  }
  return matchAt(el, tokens.length - 1);
}

function matchSelector(el, selector, scopeNode = null) {
  const parts = splitTopLevelCommas(selector);
  for (const part of parts) {
    const tokens = tokenizeComplexSelector(part);
    if (matchComplex(el, tokens, scopeNode)) return true;
  }
  return false;
}

function createMiniEnv() {
  const activeTimers = new Set();
  const activeIntervals = new Set();
  const activeRafs = new Set();
  let disposed = false;

  const rawSetTimeout = globalThis.setTimeout.bind(globalThis);
  const rawClearTimeout = globalThis.clearTimeout.bind(globalThis);
  const rawSetInterval = globalThis.setInterval.bind(globalThis);
  const rawClearInterval = globalThis.clearInterval.bind(globalThis);

  function trackedSetTimeout(fn, ms = 0, ...args) {
    if (disposed) {
      const dummy = rawSetTimeout(() => {}, 0);
      rawClearTimeout(dummy);
      return dummy;
    }
    const id = rawSetTimeout(() => {
      activeTimers.delete(id);
      if (!disposed) fn(...args);
    }, ms);
    if (typeof id?.unref === "function") id.unref();
    activeTimers.add(id);
    return id;
  }

  function trackedClearTimeout(id) {
    activeTimers.delete(id);
    rawClearTimeout(id);
  }

  function trackedSetInterval(fn, ms = 0, ...args) {
    if (disposed) {
      const dummy = rawSetTimeout(() => {}, 0);
      rawClearTimeout(dummy);
      return dummy;
    }
    const id = rawSetInterval(() => {
      if (!disposed) fn(...args);
    }, ms);
    if (typeof id?.unref === "function") id.unref();
    activeIntervals.add(id);
    return id;
  }

  function trackedClearInterval(id) {
    activeIntervals.delete(id);
    rawClearInterval(id);
  }

  function trackedRAF(cb) {
    if (disposed) return 0;
    const id = rawSetTimeout(() => {
      activeRafs.delete(id);
      if (!disposed) cb(Date.now());
    }, 0);
    if (typeof id?.unref === "function") id.unref();
    activeRafs.add(id);
    return id;
  }

  function trackedCancelRAF(id) {
    activeRafs.delete(id);
    rawClearTimeout(id);
  }

  class MiniEvent {
    constructor(type, init = {}) {
      this.type = String(type);
      this.bubbles = Boolean(init.bubbles);
      this.cancelable = Boolean(init.cancelable);
      this.defaultPrevented = false;
      this.cancelBubble = false;
      this._immediateStopped = false;
      this.target = null;
      this.currentTarget = null;
      this.eventPhase = 0;
      this.timeStamp = Date.now();
      this.isTrusted = false;
    }
    preventDefault() {
      if (this.cancelable) this.defaultPrevented = true;
    }
    stopPropagation() {
      this.cancelBubble = true;
    }
    stopImmediatePropagation() {
      this.cancelBubble = true;
      this._immediateStopped = true;
    }
  }

  class MiniKeyboardEvent extends MiniEvent {
    constructor(type, init = {}) {
      super(type, init);
      this.key = init.key ?? "";
      this.code = init.code ?? "";
      this.shiftKey = Boolean(init.shiftKey);
      this.ctrlKey = Boolean(init.ctrlKey);
      this.altKey = Boolean(init.altKey);
      this.metaKey = Boolean(init.metaKey);
      this.isComposing = Boolean(init.isComposing);
    }
  }

  class MiniMouseEvent extends MiniEvent {
    constructor(type, init = {}) {
      super(type, init);
      this.button = init.button ?? 0;
      this.buttons = init.buttons ?? 0;
      this.clientX = init.clientX ?? 0;
      this.clientY = init.clientY ?? 0;
    }
  }

  class MiniFocusEvent extends MiniEvent {
    constructor(type, init = {}) {
      super(type, init);
      this.relatedTarget = init.relatedTarget ?? null;
    }
  }

  function attachEventMethods(proto) {
    proto.addEventListener = function (type, listener, options) {
      if (!listener) return;
      const capture = typeof options === "boolean" ? options : Boolean(options?.capture);
      if (!this._listeners) this._listeners = new Map();
      let list = this._listeners.get(type);
      if (!list) {
        list = [];
        this._listeners.set(type, list);
      }
      if (!list.some((item) => item.listener === listener && item.capture === capture)) {
        list.push({ listener, capture });
      }
    };
    proto.removeEventListener = function (type, listener, options) {
      const capture = typeof options === "boolean" ? options : Boolean(options?.capture);
      const list = this._listeners?.get(type);
      if (!list) return;
      const idx = list.findIndex((item) => item.listener === listener && item.capture === capture);
      if (idx !== -1) list.splice(idx, 1);
    };
    proto.dispatchEvent = function (event) {
      event.target = this;
      const pathNodes = [];
      let cur = this;
      while (cur) {
        pathNodes.push(cur);
        cur = cur.parentNode || null;
      }
      if (this !== win && !pathNodes.includes(win)) {
        pathNodes.push(win);
      }
      const invokeOn = (node, phase, wantCapture) => {
        const list = node._listeners?.get(event.type);
        if (!list || list.length === 0) return;
        const snapshot = list.slice();
        for (const entry of snapshot) {
          if (wantCapture !== null && entry.capture !== wantCapture) continue;
          event.currentTarget = node;
          event.eventPhase = phase;
          if (typeof entry.listener === "function") {
            entry.listener.call(node, event);
          } else if (typeof entry.listener?.handleEvent === "function") {
            entry.listener.handleEvent(event);
          }
          if (event._immediateStopped) return;
        }
      };

      // Capture phase (window -> parent)
      for (let i = pathNodes.length - 1; i >= 1; i--) {
        invokeOn(pathNodes[i], 1, true);
        if (event.cancelBubble || event._immediateStopped) {
          event.currentTarget = null;
          return !event.defaultPrevented;
        }
      }
      // Target phase
      invokeOn(pathNodes[0], 2, null);
      if (event.cancelBubble || event._immediateStopped || !event.bubbles) {
        event.currentTarget = null;
        return !event.defaultPrevented;
      }
      // Bubble phase (parent -> window)
      for (let i = 1; i < pathNodes.length; i++) {
        invokeOn(pathNodes[i], 3, false);
        if (event.cancelBubble || event._immediateStopped) break;
      }
      event.currentTarget = null;
      return !event.defaultPrevented;
    };
  }

  class MiniNode {
    constructor(nodeType, ownerDocument) {
      this.nodeType = nodeType;
      this.ownerDocument = ownerDocument;
      this.parentNode = null;
      this.childNodes = [];
      this._listeners = new Map();
    }
    get firstChild() {
      return this.childNodes[0] ?? null;
    }
    get lastChild() {
      return this.childNodes[this.childNodes.length - 1] ?? null;
    }
    get nextSibling() {
      if (!this.parentNode) return null;
      const idx = this.parentNode.childNodes.indexOf(this);
      return idx >= 0 ? (this.parentNode.childNodes[idx + 1] ?? null) : null;
    }
    get previousSibling() {
      if (!this.parentNode) return null;
      const idx = this.parentNode.childNodes.indexOf(this);
      return idx > 0 ? this.parentNode.childNodes[idx - 1] : null;
    }
    get parentElement() {
      return this.parentNode && this.parentNode.nodeType === 1 ? this.parentNode : null;
    }
    get isConnected() {
      let cur = this;
      while (cur) {
        if (cur === doc || cur === doc.documentElement) return true;
        cur = cur.parentNode;
      }
      return false;
    }
    contains(other) {
      let cur = other;
      while (cur) {
        if (cur === this) return true;
        cur = cur.parentNode;
      }
      return false;
    }
    appendChild(child) {
      return this.insertBefore(child, null);
    }
    insertBefore(child, refNode) {
      if (!child) return child;
      if (child.nodeType === 11) {
        const fragChildren = child.childNodes.slice();
        for (const fc of fragChildren) {
          this.insertBefore(fc, refNode);
        }
        return child;
      }
      if (child.parentNode) {
        child.parentNode.removeChild(child);
      }
      child.parentNode = this;
      if (refNode) {
        const idx = this.childNodes.indexOf(refNode);
        if (idx !== -1) {
          this.childNodes.splice(idx, 0, child);
          return child;
        }
      }
      this.childNodes.push(child);
      return child;
    }
    removeChild(child) {
      const idx = this.childNodes.indexOf(child);
      if (idx !== -1) {
        this.childNodes.splice(idx, 1);
        child.parentNode = null;
      }
      return child;
    }
    replaceChild(newChild, oldChild) {
      this.insertBefore(newChild, oldChild);
      this.removeChild(oldChild);
      return oldChild;
    }
    remove() {
      if (this.parentNode) this.parentNode.removeChild(this);
    }
    get textContent() {
      if (this.nodeType === 3 || this.nodeType === 8) return this._data ?? "";
      return this.childNodes.map((c) => c.textContent).join("");
    }
    set textContent(val) {
      const s = val == null ? "" : String(val);
      if (this.nodeType === 3 || this.nodeType === 8) {
        this._data = s;
        return;
      }
      while (this.childNodes.length > 0) {
        this.removeChild(this.childNodes[0]);
      }
      if (s !== "") {
        this.appendChild(new MiniText(s, this.ownerDocument || doc));
      }
    }
  }
  attachEventMethods(MiniNode.prototype);

  class MiniText extends MiniNode {
    constructor(data, ownerDoc) {
      super(3, ownerDoc);
      this.nodeName = "#text";
      this._data = String(data ?? "");
    }
    get data() {
      return this._data;
    }
    set data(v) {
      this._data = String(v ?? "");
    }
    get nodeValue() {
      return this._data;
    }
    set nodeValue(v) {
      this._data = String(v ?? "");
    }
  }

  class MiniComment extends MiniNode {
    constructor(data, ownerDoc) {
      super(8, ownerDoc);
      this.nodeName = "#comment";
      this._data = String(data ?? "");
    }
    get data() {
      return this._data;
    }
    set data(v) {
      this._data = String(v ?? "");
    }
  }

  class MiniDocumentFragment extends MiniNode {
    constructor(ownerDoc) {
      super(11, ownerDoc);
      this.nodeName = "#document-fragment";
    }
  }

  function createStyleObj(el) {
    const map = new Map();
    const api = {
      setProperty(prop, val) {
        const k = String(prop).toLowerCase();
        if (val == null || val === "") map.delete(k);
        else map.set(k, String(val));
      },
      removeProperty(prop) {
        const k = String(prop).toLowerCase();
        const old = map.get(k) ?? "";
        map.delete(k);
        return old;
      },
      getPropertyValue(prop) {
        return map.get(String(prop).toLowerCase()) ?? "";
      },
    };
    return new Proxy(api, {
      get(target, prop) {
        if (prop in target) return target[prop];
        if (prop === "cssText") {
          return Array.from(map.entries())
            .map(([k, v]) => `${k}: ${v}`)
            .join("; ");
        }
        if (typeof prop === "string") {
          const kebab = prop.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);
          return map.get(kebab) ?? map.get(prop.toLowerCase()) ?? "";
        }
        return undefined;
      },
      set(_target, prop, val) {
        if (prop === "cssText") {
          map.clear();
          String(val || "")
            .split(";")
            .forEach((decl) => {
              const idx = decl.indexOf(":");
              if (idx !== -1) {
                const k = decl.slice(0, idx).trim().toLowerCase();
                const v = decl.slice(idx + 1).trim();
                if (k && v) map.set(k, v);
              }
            });
          return true;
        }
        if (typeof prop === "string") {
          const kebab = prop.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);
          if (val == null || val === "") map.delete(kebab);
          else map.set(kebab, String(val));
        }
        return true;
      },
    });
  }

  class MiniElement extends MiniNode {
    constructor(tagName, ownerDoc, namespaceURI = "http://www.w3.org/1999/xhtml") {
      super(1, ownerDoc);
      this.namespaceURI = namespaceURI;
      this.tagName =
        namespaceURI === "http://www.w3.org/1999/xhtml"
          ? String(tagName).toUpperCase()
          : String(tagName);
      this.nodeName = this.tagName;
      this._attrs = new Map();
      this.style = createStyleObj(this);
      this.scrollTop = 0;
      this.scrollHeight = 1000;
      this.clientHeight = 500;
      this.classList = {
        add: (...cls) => {
          const set = new Set(this.className.split(/\s+/).filter(Boolean));
          for (const c of cls) if (c) set.add(c);
          this._attrs.set("class", Array.from(set).join(" "));
        },
        remove: (...cls) => {
          const set = new Set(this.className.split(/\s+/).filter(Boolean));
          for (const c of cls) set.delete(c);
          const v = Array.from(set).join(" ");
          if (v) this._attrs.set("class", v);
          else this._attrs.delete("class");
        },
        contains: (c) => this.className.split(/\s+/).filter(Boolean).includes(c),
        toggle: (c, force) => {
          const has = this.classList.contains(c);
          const want = force !== undefined ? Boolean(force) : !has;
          if (want) this.classList.add(c);
          else this.classList.remove(c);
          return want;
        },
      };
    }
    get id() {
      return this._attrs.get("id") ?? "";
    }
    set id(v) {
      this.setAttribute("id", v);
    }
    get className() {
      return this._attrs.get("class") ?? "";
    }
    set className(v) {
      if (v === undefined || v === null || v === "") this._attrs.delete("class");
      else this._attrs.set("class", String(v));
    }
    get children() {
      return this.childNodes.filter((c) => c.nodeType === 1);
    }
    getAttribute(name) {
      const k = String(name).toLowerCase();
      return this._attrs.has(k) ? this._attrs.get(k) : null;
    }
    setAttribute(name, val) {
      const k = String(name).toLowerCase();
      const s = val == null ? "" : String(val);
      this._attrs.set(k, s);
      if (k === "value" && "_rawValue" in this) this._rawValue = s;
      if (k === "checked" && "_rawChecked" in this) this._rawChecked = true;
    }
    removeAttribute(name) {
      const k = String(name).toLowerCase();
      this._attrs.delete(k);
      if (k === "checked" && "_rawChecked" in this) this._rawChecked = false;
    }
    hasAttribute(name) {
      return this._attrs.has(String(name).toLowerCase());
    }
    get attributes() {
      return Array.from(this._attrs.entries()).map(([name, value]) => ({ name, value }));
    }
    set innerHTML(html) {
      while (this.childNodes.length > 0) this.removeChild(this.childNodes[0]);
      if (html && String(html).length > 0) {
        // Minimal text fallback when React sets innerHTML="" on mount
        const stripped = String(html).replace(/<[^>]*>/g, "");
        if (stripped) this.appendChild(new MiniText(stripped, this.ownerDocument || doc));
      }
    }
    get innerHTML() {
      return this.childNodes
        .map((c) => {
          if (c.nodeType === 3) return c.data;
          if (c.nodeType === 1) return `<${c.tagName.toLowerCase()}>${c.innerHTML}</${c.tagName.toLowerCase()}>`;
          return "";
        })
        .join("");
    }
    matches(selector) {
      return matchSelector(this, selector, this);
    }
    closest(selector) {
      let cur = this;
      while (cur && cur.nodeType === 1) {
        if (matchSelector(cur, selector, this)) return cur;
        cur = cur.parentNode;
      }
      return null;
    }
    querySelectorAll(selector) {
      const results = [];
      const walk = (node) => {
        for (const ch of node.childNodes) {
          if (ch.nodeType === 1) {
            if (matchSelector(ch, selector, this)) results.push(ch);
            walk(ch);
          }
        }
      };
      walk(this);
      return results;
    }
    querySelector(selector) {
      return this.querySelectorAll(selector)[0] ?? null;
    }
    getElementsByTagName(tag) {
      const t = String(tag).toLowerCase();
      return this.querySelectorAll(t === "*" ? "*" : t);
    }
    getElementsByClassName(cls) {
      return this.querySelectorAll(`.${String(cls).trim()}`);
    }
    getBoundingClientRect() {
      return { left: 0, top: 0, right: 100, bottom: 24, width: 100, height: 24 };
    }
    getClientRects() {
      let cur = this;
      while (cur && cur.nodeType === 1) {
        if (cur.style.display === "none" || cur.hasAttribute("hidden")) return [];
        cur = cur.parentNode;
      }
      return [this.getBoundingClientRect()];
    }
    scrollIntoView() {}
  }

  class MiniHTMLElement extends MiniElement {
    constructor(tagName, ownerDoc) {
      super(tagName, ownerDoc);
      this.oninput = null;
      this.onchange = null;
    }
    get disabled() {
      return this.hasAttribute("disabled");
    }
    set disabled(v) {
      if (v) this.setAttribute("disabled", "");
      else this.removeAttribute("disabled");
    }
    get tabIndex() {
      const attr = this.getAttribute("tabindex");
      if (attr !== null && /^-?\d+$/.test(attr)) return Number(attr);
      const t = this.tagName.toLowerCase();
      if (["a", "button", "input", "select", "textarea", "summary"].includes(t)) return 0;
      return -1;
    }
    set tabIndex(v) {
      this.setAttribute("tabindex", String(v));
    }
    get offsetParent() {
      let cur = this;
      while (cur && cur.nodeType === 1) {
        if (cur.style.display === "none" || cur.hasAttribute("hidden")) return null;
        cur = cur.parentNode;
      }
      return doc.body;
    }
    focus() {
      if (this.disabled) return;
      if (doc.activeElement === this) return;
      const prev = doc.activeElement;
      if (prev && prev !== this) {
        prev.dispatchEvent(new MiniFocusEvent("focusout", { bubbles: true, relatedTarget: this }));
        prev.dispatchEvent(new MiniFocusEvent("blur", { bubbles: false, relatedTarget: this }));
      }
      doc.activeElement = this;
      this.dispatchEvent(new MiniFocusEvent("focusin", { bubbles: true, relatedTarget: prev }));
      this.dispatchEvent(new MiniFocusEvent("focus", { bubbles: false, relatedTarget: prev }));
    }
    blur() {
      if (doc.activeElement !== this) return;
      doc.activeElement = doc.body;
      this.dispatchEvent(new MiniFocusEvent("focusout", { bubbles: true, relatedTarget: doc.body }));
      this.dispatchEvent(new MiniFocusEvent("blur", { bubbles: false, relatedTarget: doc.body }));
    }
    click() {
      if (this.disabled) return;
      const md = new MiniMouseEvent("mousedown", { bubbles: true, cancelable: true });
      this.dispatchEvent(md);
      if (!md.defaultPrevented && ["input", "textarea", "select", "button", "a"].includes(this.tagName.toLowerCase())) {
        this.focus();
      }
      this.dispatchEvent(new MiniMouseEvent("mouseup", { bubbles: true, cancelable: true }));
      if (this.tagName.toLowerCase() === "input" && this.type === "checkbox") {
        this._rawChecked = !this._rawChecked;
      }
      this.dispatchEvent(new MiniMouseEvent("click", { bubbles: true, cancelable: true }));
    }
  }

  class MiniHTMLInputElement extends MiniHTMLElement {
    constructor(ownerDoc) {
      super("input", ownerDoc);
      this._rawValue = "";
      this._rawChecked = false;
    }
    get type() {
      return (this.getAttribute("type") ?? "text").toLowerCase();
    }
    set type(v) {
      this.setAttribute("type", String(v));
    }
  }
  Object.defineProperty(MiniHTMLInputElement.prototype, "value", {
    configurable: true,
    enumerable: true,
    get() {
      return this._rawValue;
    },
    set(v) {
      this._rawValue = v == null ? "" : String(v);
    },
  });
  Object.defineProperty(MiniHTMLInputElement.prototype, "checked", {
    configurable: true,
    enumerable: true,
    get() {
      return this._rawChecked;
    },
    set(v) {
      this._rawChecked = Boolean(v);
    },
  });

  class MiniHTMLTextAreaElement extends MiniHTMLElement {
    constructor(ownerDoc) {
      super("textarea", ownerDoc);
      this._rawValue = "";
    }
  }
  Object.defineProperty(MiniHTMLTextAreaElement.prototype, "value", {
    configurable: true,
    enumerable: true,
    get() {
      return this._rawValue;
    },
    set(v) {
      this._rawValue = v == null ? "" : String(v);
    },
  });

  class MiniHTMLSelectElement extends MiniHTMLElement {
    constructor(ownerDoc) {
      super("select", ownerDoc);
    }
    get options() {
      return this.querySelectorAll("option");
    }
  }
  Object.defineProperty(MiniHTMLSelectElement.prototype, "value", {
    configurable: true,
    enumerable: true,
    get() {
      const opts = this.options;
      const sel = opts.find((o) => o.selected || o.hasAttribute("selected"));
      return sel ? sel.value : (opts[0]?.value ?? "");
    },
    set(v) {
      const s = v == null ? "" : String(v);
      for (const o of this.options) {
        o.selected = o.value === s;
        if (o.value === s) o.setAttribute("selected", "");
        else o.removeAttribute("selected");
      }
    },
  });

  class MiniHTMLOptionElement extends MiniHTMLElement {
    constructor(ownerDoc) {
      super("option", ownerDoc);
      this._rawSelected = false;
    }
    get value() {
      return this.hasAttribute("value") ? this.getAttribute("value") : this.textContent;
    }
    set value(v) {
      this.setAttribute("value", v);
    }
    get selected() {
      return this._rawSelected;
    }
    set selected(v) {
      this._rawSelected = Boolean(v);
    }
  }

  class MiniHTMLButtonElement extends MiniHTMLElement {
    constructor(ownerDoc) {
      super("button", ownerDoc);
    }
  }

  class MiniHTMLDetailsElement extends MiniHTMLElement {
    constructor(ownerDoc) {
      super("details", ownerDoc);
    }
    get open() {
      return this.hasAttribute("open");
    }
    set open(v) {
      if (v) this.setAttribute("open", "");
      else this.removeAttribute("open");
    }
  }

  class MiniHTMLIFrameElement extends MiniHTMLElement {
    constructor(ownerDoc) {
      super("iframe", ownerDoc);
    }
  }

  class MiniDocument extends MiniNode {
    constructor() {
      super(9, null);
      this.ownerDocument = this;
      this.nodeName = "#document";
      this.documentElement = new MiniHTMLElement("html", this);
      this.documentElement.tagName = "HTML";
      this.documentElement.nodeName = "HTML";
      this.head = new MiniHTMLElement("head", this);
      this.body = new MiniHTMLElement("body", this);
      this.documentElement.appendChild(this.head);
      this.documentElement.appendChild(this.body);
      this.appendChild(this.documentElement);
      this.activeElement = this.body;
      this.oninput = null;
      this.onchange = null;
      this.onselectionchange = null;
      this.getSelection = () => ({ rangeCount: 0, removeAllRanges() {}, addRange() {} });
    }
    createElement(tag) {
      const t = String(tag).toLowerCase();
      if (t === "input") return new MiniHTMLInputElement(this);
      if (t === "textarea") return new MiniHTMLTextAreaElement(this);
      if (t === "select") return new MiniHTMLSelectElement(this);
      if (t === "option") return new MiniHTMLOptionElement(this);
      if (t === "button") return new MiniHTMLButtonElement(this);
      if (t === "details") return new MiniHTMLDetailsElement(this);
      if (t === "iframe") return new MiniHTMLIFrameElement(this);
      return new MiniHTMLElement(t, this);
    }
    createElementNS(ns, tag) {
      if (ns === "http://www.w3.org/1999/xhtml") return this.createElement(tag);
      return new MiniElement(tag, this, ns);
    }
    createTextNode(data) {
      return new MiniText(data, this);
    }
    createComment(data) {
      return new MiniComment(data, this);
    }
    createDocumentFragment() {
      return new MiniDocumentFragment(this);
    }
    querySelector(sel) {
      return this.documentElement.querySelector(sel);
    }
    querySelectorAll(sel) {
      return this.documentElement.querySelectorAll(sel);
    }
    getElementById(id) {
      return this.querySelector(`#${id}`);
    }
    getElementsByTagName(tag) {
      return this.documentElement.getElementsByTagName(tag);
    }
    getElementsByClassName(cls) {
      return this.documentElement.getElementsByClassName(cls);
    }
  }

  const doc = new MiniDocument();
  const storageMap = new Map();
  const localStorageObj = {
    getItem: (k) => (storageMap.has(String(k)) ? storageMap.get(String(k)) : null),
    setItem: (k, v) => {
      storageMap.set(String(k), String(v));
    },
    removeItem: (k) => {
      storageMap.delete(String(k));
    },
    clear: () => {
      storageMap.clear();
    },
    get length() {
      return storageMap.size;
    },
  };

  const clipboardWrites = [];
  const win = {
    document: doc,
    location: { protocol: "http:", href: "http://localhost/", origin: "http://localhost" },
    navigator: {
      userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
      clipboard: {
        writeText: async (txt) => {
          clipboardWrites.push(String(txt));
        },
      },
    },
    localStorage: localStorageObj,
    crypto: webcrypto,
    Node: MiniNode,
    Element: MiniElement,
    HTMLElement: MiniHTMLElement,
    HTMLInputElement: MiniHTMLInputElement,
    HTMLTextAreaElement: MiniHTMLTextAreaElement,
    HTMLSelectElement: MiniHTMLSelectElement,
    HTMLOptionElement: MiniHTMLOptionElement,
    HTMLButtonElement: MiniHTMLButtonElement,
    HTMLDetailsElement: MiniHTMLDetailsElement,
    HTMLIFrameElement: MiniHTMLIFrameElement,
    Event: MiniEvent,
    KeyboardEvent: MiniKeyboardEvent,
    MouseEvent: MiniMouseEvent,
    FocusEvent: MiniFocusEvent,
    setTimeout: trackedSetTimeout,
    clearTimeout: trackedClearTimeout,
    setInterval: trackedSetInterval,
    clearInterval: trackedClearInterval,
    requestAnimationFrame: trackedRAF,
    cancelAnimationFrame: trackedCancelRAF,
    getSelection: () => ({ rangeCount: 0, removeAllRanges() {}, addRange() {} }),
    getComputedStyle: (el) => {
      const z = STYLES_AST.resolveOverlayZIndex(el);
      return {
        zIndex: z || el.style?.zIndex || "",
        display: el.style?.display || "block",
        overflowY: el.style?.overflowY || "visible",
        getPropertyValue: (p) => {
          if (p === "z-index") return z || el.style?.zIndex || "";
          return el.style?.getPropertyValue?.(p) ?? "";
        },
      };
    },
    _listeners: new Map(),
  };
  win.window = win;
  win.self = win;
  win.globalThis = win;
  attachEventMethods(win);
  doc.defaultView = win;

  const prevGlobals = {
    window: globalThis.window,
    document: globalThis.document,
    navigator: globalThis.navigator,
    location: globalThis.location,
    localStorage: globalThis.localStorage,
    Node: globalThis.Node,
    Element: globalThis.Element,
    HTMLElement: globalThis.HTMLElement,
    HTMLInputElement: globalThis.HTMLInputElement,
    HTMLTextAreaElement: globalThis.HTMLTextAreaElement,
    HTMLSelectElement: globalThis.HTMLSelectElement,
    HTMLOptionElement: globalThis.HTMLOptionElement,
    HTMLButtonElement: globalThis.HTMLButtonElement,
    HTMLDetailsElement: globalThis.HTMLDetailsElement,
    HTMLIFrameElement: globalThis.HTMLIFrameElement,
    Event: globalThis.Event,
    KeyboardEvent: globalThis.KeyboardEvent,
    MouseEvent: globalThis.MouseEvent,
    FocusEvent: globalThis.FocusEvent,
    getComputedStyle: globalThis.getComputedStyle,
    requestAnimationFrame: globalThis.requestAnimationFrame,
    cancelAnimationFrame: globalThis.cancelAnimationFrame,
    setTimeout: globalThis.setTimeout,
    clearTimeout: globalThis.clearTimeout,
    setInterval: globalThis.setInterval,
    clearInterval: globalThis.clearInterval,
    IS_REACT_ACT_ENVIRONMENT: globalThis.IS_REACT_ACT_ENVIRONMENT,
  };

  function installGlobals() {
    globalThis.window = win;
    globalThis.document = doc;
    globalThis.location = win.location;
    globalThis.localStorage = localStorageObj;
    try {
      Object.defineProperty(globalThis, "navigator", {
        configurable: true,
        writable: true,
        value: win.navigator,
      });
    } catch {
      // Ignore if non-configurable
    }
    globalThis.Node = MiniNode;
    globalThis.Element = MiniElement;
    globalThis.HTMLElement = MiniHTMLElement;
    globalThis.HTMLInputElement = MiniHTMLInputElement;
    globalThis.HTMLTextAreaElement = MiniHTMLTextAreaElement;
    globalThis.HTMLSelectElement = MiniHTMLSelectElement;
    globalThis.HTMLOptionElement = MiniHTMLOptionElement;
    globalThis.HTMLButtonElement = MiniHTMLButtonElement;
    globalThis.HTMLDetailsElement = MiniHTMLDetailsElement;
    globalThis.HTMLIFrameElement = MiniHTMLIFrameElement;
    globalThis.Event = MiniEvent;
    globalThis.KeyboardEvent = MiniKeyboardEvent;
    globalThis.MouseEvent = MiniMouseEvent;
    globalThis.FocusEvent = MiniFocusEvent;
    globalThis.getComputedStyle = win.getComputedStyle;
    globalThis.requestAnimationFrame = trackedRAF;
    globalThis.cancelAnimationFrame = trackedCancelRAF;
    globalThis.setTimeout = trackedSetTimeout;
    globalThis.clearTimeout = trackedClearTimeout;
    globalThis.setInterval = trackedSetInterval;
    globalThis.clearInterval = trackedClearInterval;
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    if (!globalThis.crypto) globalThis.crypto = webcrypto;
  }

  function restoreGlobals() {
    disposed = true;
    for (const id of Array.from(activeTimers)) rawClearTimeout(id);
    activeTimers.clear();
    for (const id of Array.from(activeIntervals)) rawClearInterval(id);
    activeIntervals.clear();
    for (const id of Array.from(activeRafs)) rawClearTimeout(id);
    activeRafs.clear();
    storageMap.clear();
    for (const [k, v] of Object.entries(prevGlobals)) {
      if (k === "navigator") {
        try {
          Object.defineProperty(globalThis, "navigator", {
            configurable: true,
            writable: true,
            value: v,
          });
        } catch {
          // ignore
        }
      } else if (v === undefined) {
        delete globalThis[k];
      } else {
        globalThis[k] = v;
      }
    }
  }

  async function flushAnimationFrames(maxRounds = 10) {
    let rounds = 0;
    while (!disposed && activeRafs.size > 0 && rounds < maxRounds) {
      rounds += 1;
      await new Promise((resolve) => rawSetTimeout(resolve, 0));
    }
  }

  return {
    win,
    doc,
    localStorage: localStorageObj,
    clipboardWrites,
    installGlobals,
    restoreGlobals,
    flushAnimationFrames,
    MiniEvent,
    MiniKeyboardEvent,
    MiniMouseEvent,
    MiniFocusEvent,
  };
}

// ---------------------------------------------------------------------------
// 3. In-Memory esbuild Bundle of src/App.tsx
// ---------------------------------------------------------------------------
let cachedAppCode = null;
function getBundledAppCode() {
  if (cachedAppCode) return cachedAppCode;
  const result = esbuild.buildSync({
    entryPoints: [SRC_APP_PATH],
    bundle: true,
    write: false,
    format: "cjs",
    jsx: "automatic",
    platform: "browser",
    external: ["react", "react-dom", "react-dom/*", "react/*", "lucide-react", "@tauri-apps/*"],
  });
  cachedAppCode = result.outputFiles[0].text;
  return cachedAppCode;
}

function loadAppComponent() {
  const code = getBundledAppCode();
  const mod = { exports: {} };
  const fn = new Function("require", "module", "exports", "__filename", "__dirname", code);
  fn(require, mod, mod.exports, SRC_APP_PATH, path.dirname(SRC_APP_PATH));
  return mod.exports.default;
}

// ---------------------------------------------------------------------------
// 4. Fixture Builders & App Mounting Harness
// ---------------------------------------------------------------------------
function makeEnv(overrides = {}) {
  return {
    colab_bin: "colab",
    colab_ok: true,
    rclone_bin: "rclone",
    rclone_ok: true,
    rclone_conf: "/home/user/.config/rclone/rclone.conf",
    rclone_conf_ok: true,
    token_ok: true,
    adc_ok: true,
    workspace: "D:/default-ws",
    sessions_path: "/home/user/.config/colab-cli/sessions.json",
    history_dir: "/home/user/.config/colab-cli/history",
    ...overrides,
  };
}

function makeCcu(overrides = {}) {
  return {
    balance: 95.5,
    burn_rate_hourly: 1.96,
    assignments_count: 1,
    ...overrides,
  };
}

function makeKeepAlive(overrides = {}) {
  return {
    enabled: true,
    interval_seconds: 90,
    kernel_ping: true,
    state: "active",
    last_ok_at: 1732630000,
    last_attempt_at: 1732630000,
    tunnel_ok: true,
    kernel_ping_state: "ok",
    success_count: 3,
    consecutive_failures: 0,
    last_error: null,
    profile_saved: true,
    persistence_error: null,
    ...overrides,
  };
}

function makeRelay(vmName = "relay-vm", overrides = {}) {
  return {
    vm_name: vmName,
    running: true,
    round: 1,
    phase: "training",
    phase_label: "训练热备中",
    started_at: Math.floor(Date.now() / 1000) - 120,
    round_started_at: Math.floor(Date.now() / 1000) - 120,
    config: {
      gpu_type: "T4",
      high_mem: true,
      gdrive_dir: "gdrive:colab_relay",
      max_session_hours: 22.5,
      backup_interval_minutes: 30,
      total_epochs: 100,
      reset_done_flag: false,
      grab_mode: "budget",
      grab_timeout_minutes: 60,
      grab_retry_seconds: 20,
      fallback_gpus: [],
      ...(overrides.config ?? {}),
    },
    logs: ["[round 1] training started"],
    child_pid: 4321,
    training_completed: false,
    weights_downloaded: false,
    download_error: null,
    local_weights_path: null,
    last_error: null,
    grab_status: null,
    ...overrides,
  };
}

function makeLlmApi(vmName = "llm-blue", overrides = {}) {
  return {
    vm_name: vmName,
    slot: "blue",
    model_name: "Qwen/Qwen2.5-7B-Instruct-GGUF",
    engine: "llama.cpp",
    api_base_url: "http://127.0.0.1:18081/v1",
    chat_completions_url: "http://127.0.0.1:18081/v1/chat/completions",
    models_url: "http://127.0.0.1:18081/v1/models",
    gateway_base_url: "http://127.0.0.1:18080/v1",
    api_key: "sk-colab-secret-1234567890abcdef",
    local_port: 18081,
    remote_port: 8000,
    curl_example: "curl http://127.0.0.1:18081/v1/chat/completions",
    python_example: "from openai import OpenAI",
    ...overrides,
  };
}

function makeVm(overrides = {}) {
  const name = overrides.name ?? "vm-1";
  return {
    name,
    workspace: {
      name: "default-ws",
      path: "D:/default-ws",
      remote_cwd: "/content/default-ws",
      source: "default",
      active_script: "train.py",
      description: null,
      ...(overrides.workspace ?? {}),
    },
    endpoint: "m-s-ep1",
    accelerator: "T4",
    variant: "GPU",
    high_ram: true,
    on_server: true,
    managed: true,
    orphan: false,
    kernel_id: "kern-1",
    session_id: "sess-1",
    running_task: null,
    last_execution: null,
    cli_events: [],
    telemetry: {
      timestamp: 1732630000,
      uptime_seconds: 300,
      cpu: { model: "Xeon", cores: 4, percent: 18, load_avg: [0.3, 0.2, 0.1] },
      memory: { total_gb: 51, used_gb: 6, avail_gb: 45, percent: 12 },
      disk: { total_gb: 112, used_gb: 24, free_gb: 88, percent: 21 },
      gpu: { available: true, name: "Tesla T4", gpu_util_percent: 42, vram_used_mb: 4096, vram_total_mb: 15360, vram_percent: 26 },
      threads: {
        total_processes: 3,
        total_threads: 12,
        active_process: null,
        train_process: null,
        llm_process: null,
        rclone_process: null,
        processes: [],
      },
      files: { has_rclone_conf: true, has_checkpoint: false, checkpoint_size_mb: 0, has_done_flag: false, entries: [] },
      training: null,
      llm: null,
      ...(overrides.telemetry ?? {}),
    },
    telemetry_age: 2,
    telemetry_error: null,
    deploy_error: null,
    server_state_unknown: false,
    relay: null,
    busy: null,
    ssh: null,
    llm_api: null,
    keep_alive: makeKeepAlive(),
    grab_status: null,
    ...overrides,
    // Keep nested telemetry merged cleanly if provided
    ...(overrides.telemetry === null ? { telemetry: null } : {}),
  };
}

function makeOverview(overrides = {}) {
  return {
    timestamp: 1732630000,
    seq: 1,
    last_ok_timestamp: 1732630000,
    is_stale: false,
    stale_reason: null,
    server_list_known: true,
    auth_ok: true,
    error: null,
    ccu: makeCcu(),
    server_count: Array.isArray(overrides.vms) ? overrides.vms.filter((v) => v.on_server).length : 0,
    gdrive: null,
    env: makeEnv(),
    vms: [],
    recent_sessions: [],
    llm_bluegreen: null,
    ...overrides,
  };
}

let activeHarnessCleanup = null;

afterEach(async () => {
  if (activeHarnessCleanup) {
    const fn = activeHarnessCleanup;
    activeHarnessCleanup = null;
    await fn();
  }
});

async function mountApp({ initialOverview, handlers = {}, preSeedStorage = null } = {}) {
  const env = createMiniEnv();
  env.installGlobals();
  if (preSeedStorage) {
    for (const [k, v] of Object.entries(preSeedStorage)) {
      env.localStorage.setItem(k, v);
    }
  }

  const React = require("react");
  const { createRoot } = require("react-dom/client");
  const { mockIPC, mockWindows, clearMocks } = require("@tauri-apps/api/mocks");
  const { emit } = require("@tauri-apps/api/event");

  let currentOverview = initialOverview ?? makeOverview();
  const ipcCalls = [];
  const consoleErrors = [];
  const origConsoleError = console.error;
  console.error = (...args) => {
    consoleErrors.push(args.map((a) => String(a)).join(" "));
  };

  mockWindows("main");
  mockIPC(
    async (cmd, args) => {
      ipcCalls.push({ cmd, args: args ?? {} });
      if (handlers[cmd]) {
        return await handlers[cmd](args ?? {}, {
          getOverview: () => currentOverview,
          setOverview: (next) => {
            currentOverview = next;
          },
        });
      }
      if (cmd === "get_overview") return currentOverview;
      if (cmd === "refresh_now") return currentOverview;
      if (cmd === "set_script_dirty") return null;
      return null;
    },
    { shouldMockEvents: true },
  );

  const App = loadAppComponent();
  const container = env.doc.createElement("div");
  container.id = "root";
  env.doc.body.appendChild(container);
  const root = createRoot(container);

  await React.act(async () => {
    root.render(React.createElement(App));
  });
  // Flush initial async get_overview + refresh_now + effect promises
  await React.act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });

  let cleaned = false;
  async function unmount() {
    if (cleaned) return;
    cleaned = true;
    try {
      await React.act(async () => {
        root.unmount();
        for (let i = 0; i < 6; i++) await Promise.resolve();
      });
    } finally {
      console.error = origConsoleError;
      clearMocks();
      env.restoreGlobals();
    }
  }
  activeHarnessCleanup = unmount;

  async function flush(ticks = 3) {
    await React.act(async () => {
      for (let i = 0; i < ticks; i++) {
        await Promise.resolve();
      }
      await env.flushAnimationFrames();
      for (let i = 0; i < ticks; i++) {
        await Promise.resolve();
      }
    });
  }

  async function emitEvent(eventName, payload) {
    await React.act(async () => {
      await emit(eventName, payload);
      await Promise.resolve();
      await Promise.resolve();
    });
  }

  function findAll(selector, scope = container) {
    return scope.querySelectorAll(selector);
  }

  function findOne(selector, scope = container) {
    return scope.querySelector(selector);
  }

  function findByText(selector, textMatch, scope = container) {
    const all = scope.querySelectorAll(selector);
    return all.filter((el) => {
      const txt = el.textContent.trim();
      return textMatch instanceof RegExp ? textMatch.test(txt) : txt.includes(textMatch);
    });
  }

  function buttonByText(textMatch, scope = container) {
    const btns = findByText("button", textMatch, scope);
    assert.ok(btns.length > 0, `Expected button matching ${String(textMatch)}, found none`);
    return btns[0];
  }

  function hwButton(code, scope = container) {
    const btns = findAll(".hw-opt", scope).filter((el) => {
      const titleSpan = el.querySelector(".t");
      return titleSpan && titleSpan.textContent.trim() === code;
    });
    assert.ok(btns.length > 0, `Expected hardware option ${code}, found none`);
    return btns[0];
  }

  async function click(el) {
    assert.ok(el, "click() target is null/undefined");
    await React.act(async () => {
      el.click();
      await Promise.resolve();
      await Promise.resolve();
    });
  }

  async function setInputValue(el, value) {
    assert.ok(el, "setInputValue() target is null/undefined");
    await React.act(async () => {
      const proto =
        el.tagName === "TEXTAREA"
          ? env.win.HTMLTextAreaElement.prototype
          : el.tagName === "SELECT"
            ? env.win.HTMLSelectElement.prototype
            : env.win.HTMLInputElement.prototype;
      const desc = Object.getOwnPropertyDescriptor(proto, "value");
      desc.set.call(el, value);
      el.dispatchEvent(new env.MiniEvent("input", { bubbles: true, cancelable: true }));
      el.dispatchEvent(new env.MiniEvent("change", { bubbles: true, cancelable: true }));
      await Promise.resolve();
    });
  }

  async function setCheckbox(el, checked) {
    assert.ok(el, "setCheckbox() target is null/undefined");
    if (Boolean(el.checked) === Boolean(checked)) return;
    await click(el);
  }

  async function selectOption(selectEl, value) {
    assert.ok(selectEl, "selectOption() target is null/undefined");
    await React.act(async () => {
      const desc = Object.getOwnPropertyDescriptor(env.win.HTMLSelectElement.prototype, "value");
      desc.set.call(selectEl, value);
      selectEl.dispatchEvent(new env.MiniEvent("change", { bubbles: true, cancelable: true }));
      await Promise.resolve();
    });
  }

  async function pressKey(key, opts = {}) {
    await React.act(async () => {
      const target = opts.target ?? env.doc.activeElement ?? env.doc.body;
      const ev = new env.MiniKeyboardEvent("keydown", {
        key,
        bubbles: true,
        cancelable: true,
        shiftKey: Boolean(opts.shiftKey),
        ctrlKey: Boolean(opts.ctrlKey),
        altKey: Boolean(opts.altKey),
      });
      target.dispatchEvent(ev);
      await Promise.resolve();
      await Promise.resolve();
    });
  }

  async function focusEl(el) {
    await React.act(async () => {
      el.focus();
      await Promise.resolve();
    });
  }

  async function blurEl(el) {
    await React.act(async () => {
      el.blur();
      await Promise.resolve();
    });
  }

  function bodyText() {
    return container.textContent;
  }

  function toastTexts() {
    return findAll(".toast").map((t) => t.textContent.trim());
  }

  function cmds(name) {
    return ipcCalls.filter((c) => c.cmd === name);
  }

  function setOverviewSnapshot(next) {
    currentOverview = next;
  }

  return {
    env,
    doc: env.doc,
    win: env.win,
    container,
    ipcCalls,
    cmds,
    consoleErrors,
    clipboardWrites: env.clipboardWrites,
    emitEvent,
    setOverviewSnapshot,
    flush,
    unmount,
    findAll,
    findOne,
    findByText,
    buttonByText,
    hwButton,
    click,
    setInputValue,
    setCheckbox,
    selectOption,
    pressKey,
    focusEl,
    blurEl,
    bodyText,
    toastTexts,
  };
}

// ===========================================================================
// TIER 1 — Category-Partition Feature Coverage (11 tests)
// ===========================================================================

test("test_t1_f1_vmcard_keepalive_bar_renders_policy_and_presets", async () => {
  const vmActive = makeVm({
    name: "ka-active-vm",
    keep_alive: makeKeepAlive({
      enabled: true,
      interval_seconds: 180,
      kernel_ping: false,
      state: "active",
      kernel_ping_state: "disabled",
      success_count: 5,
    }),
  });
  const vmPersistErr = makeVm({
    name: "ka-persist-err-vm",
    keep_alive: makeKeepAlive({
      enabled: true,
      interval_seconds: 90,
      kernel_ping: true,
      state: "active",
      profile_saved: false,
      persistence_error: "pilot_session_profiles.json 只读不可写",
    }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({ vms: [vmActive, vmPersistErr], server_count: 2 }),
  });

  const cards = h.findAll(".vm-card");
  assert.equal(cards.length, 2);

  // Card 1: kernel_ping=false renders 连接检查运行中 (180s), Kernel 脉冲已关闭, presets 60/90/180/300, numeric input 180
  const bar1 = h.findOne(".ka-bar", cards[0]);
  assert.ok(bar1.textContent.includes("连接检查运行中 (180s)"));
  assert.ok(bar1.textContent.includes("Kernel 脉冲已关闭"));
  const input1 = h.findOne('input[aria-label="连接检查间隔"]', bar1);
  assert.ok(input1, "KeepAliveBar must render numeric input for #98");
  assert.equal(input1.getAttribute("type"), "number");
  assert.equal(input1.getAttribute("min"), "30");
  assert.equal(input1.getAttribute("max"), "600");
  assert.equal(input1.value, "180");
  const presets1 = h.findAll(".ka-preset-pill", bar1).map((b) => b.textContent.trim());
  assert.deepEqual(presets1, ["60s", "90s", "180s", "300s"]);

  // Card 2: persistence_error surfaced via .ka-persistence-alert + retry button (#97)
  const bar2 = h.findOne(".ka-bar", cards[1]);
  const alert2 = h.findOne(".ka-persistence-alert", bar2);
  assert.ok(alert2, "KeepAliveBar must render persistence alert for #97");
  assert.ok(alert2.textContent.includes("pilot_session_profiles.json 只读不可写"));
  assert.ok(h.buttonByText("重试保存策略", alert2));
  assert.equal(h.consoleErrors.filter((e) => e.includes("[ErrorBoundary")).length, 0);
});

test("test_t1_f1_ping_keepalive_preserves_paused_policy", async () => {
  const pausedVm = makeVm({
    name: "ka-paused-vm",
    keep_alive: makeKeepAlive({
      enabled: false,
      interval_seconds: 120,
      kernel_ping: true,
      state: "paused",
      last_ok_at: null,
      last_attempt_at: null,
      tunnel_ok: false,
      kernel_ping_state: "disabled",
      success_count: 0,
    }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({ vms: [pausedVm], server_count: 1 }),
    handlers: {
      ping_vm_keep_alive: async ({ name }) => ({
        enabled: false,
        interval_seconds: 120,
        kernel_ping: true,
        state: "paused",
        last_ok_at: 1732630500,
        last_attempt_at: 1732630500,
        tunnel_ok: true,
        kernel_ping_state: "ok",
        success_count: 0,
        consecutive_failures: 0,
        last_error: null,
        profile_saved: true,
      }),
    },
  });

  const bar = h.findOne(".ka-bar");
  assert.ok(bar.textContent.includes("保活未开启"));
  // On paused VM with kernel_ping=true, button shows 单次保活 (not turning on periodic policy)
  const pingBtn = h.buttonByText("单次保活", bar);
  await h.click(pingBtn);

  assert.equal(h.cmds("ping_vm_keep_alive").length, 1);
  assert.equal(h.cmds("ping_vm_keep_alive")[0].args.name, "ka-paused-vm");
  // Crucial #91 invariant: single ping NEVER invokes set_vm_keep_alive
  assert.equal(h.cmds("set_vm_keep_alive").length, 0);

  const toasts = h.toastTexts();
  assert.ok(
    toasts.some((t) => t.includes("单次检查正常") && t.includes("自动保活仍为关闭状态")),
    `Expected paused single-shot toast, got: ${JSON.stringify(toasts)}`,
  );
});

test("test_t1_f2_relay_cpu_preservation_and_custom_workspace_routing", async () => {
  let latestCreatedReq = null;
  const h = await mountApp({
    initialOverview: makeOverview({
      vms: [
        makeVm({
          name: "relay-cpu-ws",
          accelerator: "CPU",
          high_ram: false,
          workspace: {
            name: "ws-b",
            path: "D:/ws-b",
            remote_cwd: "/content/ws-b",
            source: "custom",
            active_script: "train.py",
            description: null,
          },
          relay: makeRelay("relay-cpu-ws", {
            running: false,
            phase: "idle",
            phase_label: "未开启接力",
            config: {
              gpu_type: "CPU",
              high_mem: false,
              gdrive_dir: "gdrive:colab_relay/relay-cpu-ws",
              max_session_hours: 12,
              backup_interval_minutes: 30,
              total_epochs: 50,
              reset_done_flag: false,
            },
          }),
        }),
      ],
      server_count: 1,
    }),
    handlers: {
      create_vm: async ({ req }) => {
        latestCreatedReq = req;
        return req.name;
      },
      read_workspace_files: async ({ workspacePath, vmName }) => ({
        workspace: workspacePath ?? "D:/default-ws",
        train_py: `# script for ${vmName} at ${workspacePath}`,
        requirements_txt: "torch\n",
      }),
    },
  });

  // 1) Create VM in relay mode with CPU and custom workspace D:/ws-b (#42, #52)
  await h.click(h.buttonByText("开机 / 新建实例"));
  const modal = h.findOne(".modal");
  await h.click(h.buttonByText("24h+ 自动接力训练", modal));
  await h.click(h.hwButton("CPU", modal));
  await h.click(h.buttonByText("更多 Colab 运行时与环境配置", modal));
  const wsInput = h.findOne('input[placeholder="默认绑定当前项目工作区目录"]', modal);
  await h.setInputValue(wsInput, "D:/ws-b");
  await h.click(h.buttonByText("开机并启动接力训练 (CPU)", modal));

  assert.ok(latestCreatedReq, "create_vm should have been called");
  assert.equal(latestCreatedReq.accelerator, "CPU");
  assert.equal(latestCreatedReq.relay_config.gpu_type, "CPU", "#52 must preserve CPU, never coerce to T4");
  assert.equal(latestCreatedReq.workspace_path, "D:/ws-b", "#42 must route custom workspace");

  // 2) Open ScriptDialog from VmCard bound to D:/ws-b (#42)
  const card = h.findOne(".vm-card");
  await h.click(h.buttonByText("编辑训练脚本", card));
  const readCalls = h.cmds("read_workspace_files");
  assert.equal(readCalls.length, 1);
  assert.equal(readCalls[0].args.workspacePath, "D:/ws-b");
  assert.equal(readCalls[0].args.vmName, "relay-cpu-ws");
  const scriptModal = h.findOne(".modal");
  assert.ok(scriptModal.textContent.includes("工作目录：D:/ws-b"));
});

test("test_t1_f2_relay_dir_isolation_and_offline_dismiss", async () => {
  const runningVm = makeVm({
    name: "trainer-alpha",
    relay: makeRelay("trainer-alpha", {
      running: true,
      config: { gdrive_dir: "gdrive:colab_relay" },
    }),
  });
  const offlineRelayVm = makeVm({
    name: "trainer-done-offline",
    on_server: false,
    endpoint: null,
    telemetry: null,
    relay: makeRelay("trainer-done-offline", {
      running: false,
      phase: "error",
      phase_label: "接力失败",
      last_error: "OOM during epoch 14",
    }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({ vms: [runningVm, offlineRelayVm], server_count: 1 }),
    handlers: {
      dismiss_relay_status: async (_args, ctx) => {
        const cur = ctx.getOverview();
        ctx.setOverview({ ...cur, vms: cur.vms.filter((v) => v.name !== "trainer-done-offline") });
        return null;
      },
    },
  });

  // Offline relay card must remain visible (#73) with error details and dismiss button
  const cards = h.findAll(".vm-card");
  assert.equal(cards.length, 2);
  const offlineCard = cards[1];
  assert.ok(offlineCard.textContent.includes("OOM during epoch 14"));
  assert.ok(offlineCard.textContent.includes("接力失败 · 实例已下线"));

  // Verify default bare gdrive_dir isolation preview & conflict detection in CreateVmDialog (#44)
  await h.click(h.buttonByText("开机 / 新建实例"));
  const modal = h.findOne(".modal");
  await h.click(h.buttonByText("24h+ 自动接力训练", modal));
  const nameInput = h.findOne('input[placeholder="例如 gpu-dev-01"]', modal);
  await h.setInputValue(nameInput, "trainer-beta");
  assert.ok(
    modal.textContent.includes("gdrive:colab_relay/trainer-beta"),
    "#44 should show isolated subdir gdrive:colab_relay/trainer-beta",
  );
  // Entering explicit gdrive_dir matching running trainer-alpha triggers conflict block until explicit confirm
  const gdriveInput = h.findByText("label", "Google Drive 断点目录", modal)[0].parentNode.querySelector("input");
  await h.setInputValue(gdriveInput, "gdrive:colab_relay/trainer-alpha");
  assert.ok(modal.textContent.includes("正被运行中的接力实例 [trainer-alpha] 占用"));
  assert.equal(h.buttonByText("开机并启动接力训练", modal).disabled, true);
  await h.click(h.buttonByText("取消", modal));

  // Dismiss offline relay card (#73)
  await h.click(h.buttonByText("清除记录", offlineCard));
  assert.equal(h.cmds("dismiss_relay_status").length, 1);
  assert.equal(h.cmds("dismiss_relay_status")[0].args.name, "trainer-done-offline");
  assert.equal(h.findAll(".vm-card").length, 1);
});

test("test_t1_f3_llm_hardware_cuda_compat_guard", async () => {
  const h = await mountApp({
    initialOverview: makeOverview(),
    handlers: {
      create_vm: async ({ req }) => req.name,
    },
  });

  await h.click(h.buttonByText("开机 / 新建实例"));
  const modal = h.findOne(".modal");
  await h.click(h.buttonByText("部署大模型 + API / Key", modal));

  // Switch hardware to CPU while CUDA-only engine llama.cpp is selected -> must block (#68)
  await h.click(h.hwButton("CPU", modal));
  assert.ok(modal.textContent.includes("硬件不兼容"));
  assert.ok(modal.textContent.includes("依赖 NVIDIA CUDA"));
  assert.equal(h.buttonByText("开机并部署大模型 (CPU)", modal).disabled, true);

  // Switch to TPU V5E1 -> still blocked for llama.cpp / vllm
  await h.click(h.hwButton("V5E1", modal));
  const engineSelect = h.findOne("select", modal);
  await h.selectOption(engineSelect, "vllm");
  assert.equal(h.buttonByText("开机并部署大模型 (V5E1)", modal).disabled, true);

  // One-click switch to Ollama -> compatible on V5E1, submit re-enabled!
  await h.click(h.buttonByText("切换至 Ollama", modal));
  assert.equal(h.buttonByText("开机并部署大模型 (V5E1)", modal).disabled, false);

  // Switch to T4 and llama.cpp -> compatible & enabled
  await h.click(h.hwButton("T4", modal));
  await h.selectOption(engineSelect, "llama.cpp");
  assert.equal(h.buttonByText("开机并部署大模型 (T4)", modal).disabled, false);
});

test("test_t1_f3_llm_idle_tps_display_and_key_rotation_error", async () => {
  const llmVm = makeVm({
    name: "llm-node-a",
    llm_api: makeLlmApi("llm-node-a"),
    telemetry: {
      timestamp: 1732630000,
      uptime_seconds: 500,
      cpu: { model: "Xeon", cores: 4, percent: 10, load_avg: [0.1] },
      memory: { total_gb: 51, used_gb: 10, avail_gb: 41, percent: 20 },
      disk: { total_gb: 112, used_gb: 30, free_gb: 82, percent: 27 },
      gpu: { available: true, name: "Tesla T4", gpu_util_percent: 5, vram_used_mb: 5000, vram_total_mb: 15360, vram_percent: 33 },
      threads: { total_processes: 2, total_threads: 8, active_process: null, train_process: null, llm_process: null, rclone_process: null, processes: [] },
      files: { has_rclone_conf: true, has_checkpoint: false, checkpoint_size_mb: 0, has_done_flag: false, entries: [] },
      training: null,
      llm: {
        mode: "llm",
        status: "online",
        online: true,
        backend_healthy: true,
        slot: "blue",
        role: "active",
        engine: "llama.cpp",
        model_name: "Qwen/Qwen2.5-7B-Instruct-GGUF",
        instant_tps: 0,
        last_request_tps: 38,
        avg_tps_60s: 12.4,
        peak_tps: 44.2,
        ttft_ms: 210,
        active_streams: 0,
        total_requests: 4,
        total_completion_tokens: 512,
      },
    },
  });

  const h = await mountApp({
    initialOverview: makeOverview({
      vms: [llmVm],
      server_count: 1,
      llm_bluegreen: {
        active_slot: "blue",
        active_session: "llm-node-a",
        blue_session: "llm-node-a",
        model_name: "Qwen/Qwen2.5-7B-Instruct-GGUF",
      },
    }),
    handlers: {
      rotate_llm_api_key: async () => {
        throw new Error("llm_api_keys.json 写入失败: Permission denied");
      },
    },
  });

  const text = h.bodyText();
  // #65: idle instant_tps=0 keeps 0.0 tok/s (空闲), and last_request_tps=38 is labeled 上次请求: 38.0 tok/s
  assert.ok(text.includes("0.0 tok/s (空闲)"));
  assert.ok(text.includes("上次请求: 38.0 tok/s"));
  assert.ok(!text.includes("实时速率38.0"));

  // #59: rotate key error must surface error toast and NEVER claim success
  const rotateBtn = h.findOne('button[title="重新生成此虚拟机的 API Key"]');
  assert.ok(rotateBtn);
  await h.click(rotateBtn);
  assert.equal(h.cmds("rotate_llm_api_key").length, 1);
  const toasts = h.toastTexts();
  assert.ok(toasts.some((t) => t.includes("轮换 API Key 失败") && t.includes("Permission denied")));
  assert.ok(!toasts.some((t) => t.includes("已更新")));
});

test("test_t1_f4_billing_unknown_renders_unknown_never_zero_green", async () => {
  const offlineRelayVm = makeVm({
    name: "relay-offline-unknown",
    on_server: false,
    server_state_unknown: true,
    telemetry: null,
    relay: makeRelay("relay-offline-unknown", { running: false, phase: "stopped", phase_label: "已停止" }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({
      server_list_known: false,
      ccu: null,
      server_count: 0,
      error: "colab status failed",
      vms: [offlineRelayVm],
    }),
    handlers: {
      delete_vm: async () => ({
        name: "relay-offline-unknown",
        endpoint: "m-s-stale",
        verified_unassigned: false,
        remaining_endpoints: ["m-s-stale"],
        steps: ["unassign attempted"],
        ccu: null,
        error: "endpoint still active",
      }),
    },
  });

  // #56 & #78: Header must show 未知 for instances and burn rate, never green 0.00 CU/h
  const header = h.findOne("header.topbar");
  const statValues = h.findAll(".stat-value", header).map((s) => s.textContent.trim());
  assert.ok(statValues.includes("未知"), `Header stats should show 未知: ${JSON.stringify(statValues)}`);
  assert.ok(!header.textContent.includes("0.00 CU/h"), "Header must never show 0.00 CU/h when CCU missing");

  // #78: Offline relay card when server_list_known=false must say 云端计费状态未知 and NEVER 不再产生计费
  const card = h.findOne(".vm-card");
  assert.ok(card.textContent.includes("云端计费状态未知"));
  assert.ok(!card.textContent.includes("不再产生计费"));
  assert.ok(!card.textContent.includes("不再计费"));
});

test("test_t1_f4_snapshot_ordering_and_manual_stale_refresh", async () => {
  const vmLive = makeVm({ name: "live-vm-seq5" });
  const freshSnap = makeOverview({ seq: 5, timestamp: 2000, last_ok_timestamp: 2000, vms: [vmLive], server_count: 1 });
  let refreshReply = freshSnap;

  const h = await mountApp({
    initialOverview: freshSnap,
    handlers: {
      refresh_now: async () => refreshReply,
    },
  });

  assert.ok(h.bodyText().includes("live-vm-seq5"));

  // #82: Emit an older stale snapshot (seq 4, ts 1900, empty vms) -> MUST be ignored, live-vm-seq5 preserved!
  await h.emitEvent(
    "overview",
    makeOverview({
      seq: 4,
      timestamp: 1900,
      last_ok_timestamp: 1900,
      is_stale: true,
      stale_reason: "old failure",
      vms: [],
      server_count: 0,
    }),
  );
  assert.ok(h.bodyText().includes("live-vm-seq5"), "#82 older snapshot must not overwrite newer VM");

  // #57: Manual refresh returns newer seq=6 failure hitting cache -> keep VMs, show warn banner + toast + last_ok_timestamp
  refreshReply = makeOverview({
    seq: 6,
    timestamp: 2060,
    last_ok_timestamp: 1950,
    is_stale: true,
    stale_reason: "Colab API 503",
    vms: [],
    server_count: 0,
  });
  const refreshBtn = h.findOne('header.topbar button[title*="立即刷新"]');
  await h.click(refreshBtn);

  assert.ok(h.bodyText().includes("live-vm-seq5"), "Cached VMs preserved on stale failure");
  assert.ok(h.bodyText().includes("最新查询失败：Colab API 503"));
  assert.ok(h.toastTexts().some((t) => t.includes("最新查询失败") && t.includes("Colab API 503")));
});

test("test_t1_f4_create_vm_general_gdrive_optout_multiline_env_and_auth_guide", async () => {
  let createdReq = null;
  const h = await mountApp({
    initialOverview: makeOverview({
      env: makeEnv({ token_ok: false, adc_ok: false, rclone_conf_ok: false }),
    }),
    handlers: {
      create_vm: async ({ req }) => {
        createdReq = req;
        return req.name;
      },
    },
  });

  // #75: Unauthenticated banner guides `colab auth` and `colab status`, NEVER `colab auth login`
  assert.ok(h.bodyText().includes("colab auth"));
  assert.ok(h.bodyText().includes("colab status"));
  assert.ok(!h.bodyText().includes("colab auth login"));

  await h.click(h.buttonByText("开机 / 新建实例"));
  const modal = h.findOne(".modal");

  // #43: Selecting ADC when token_ok=false blocks creation with danger alert
  await h.click(h.buttonByText("ADC 凭证", modal));
  assert.ok(modal.textContent.includes("已阻止分配"));
  assert.equal(h.buttonByText("立即开机 (T4)", modal).disabled, true);
  assert.ok(!modal.textContent.includes("colab auth login"));

  // Switch back to OAuth2 -> allowed in general mode even when rclone_conf_ok=false (#67)
  await h.click(h.buttonByText("OAuth2 账号", modal));
  assert.ok(!modal.textContent.includes("当前配置依赖 Google Drive"), "#67 general mode does not require rclone");

  // Expand advanced settings and check multiline env_vars textarea (#69) & sync_rclone default false (#67)
  await h.click(h.buttonByText("更多 Colab 运行时与环境配置", modal));
  const syncCheckbox = h
    .findAll("label.adv-check-item", modal)
    .find((l) => l.textContent.includes("rclone.conf"))
    .querySelector("input");
  assert.equal(syncCheckbox.checked, false, "#67 sync_rclone must default false in general mode");

  const envTa = h.findOne('textarea[placeholder*="HF_TOKEN=hf_xxx"]', modal);
  assert.ok(envTa, "#69 env_vars must be a <textarea> supporting multiline input");
  await h.setInputValue(envTa, "FOO=bar\nBAZ=qux");

  await h.click(h.buttonByText("立即开机 (T4)", modal));
  assert.ok(createdReq);
  assert.equal(createdReq.env_config.sync_rclone, false);
  assert.equal(createdReq.env_config.env_vars, "FOO=bar\nBAZ=qux");
});

test("test_t1_f4_script_dialog_read_failure_and_dirty_esc_guard", async () => {
  let mode = "ok";
  const h = await mountApp({
    initialOverview: makeOverview(),
    handlers: {
      read_workspace_files: async () => {
        if (mode === "fail") throw new Error("train.py UTF-8 解码失败");
        return { workspace: "D:/default-ws", train_py: "print('secret_v1')", requirements_txt: "torch\n" };
      },
      save_workspace_files: async () => null,
    },
  });

  // 1) Open cleanly and load secret_v1, then close
  await h.click(h.buttonByText("配置训练脚本"));
  let modal = h.findOne(".modal");
  const textareas1 = h.findAll("textarea", modal);
  assert.equal(textareas1[0].value, "print('secret_v1')");
  await h.click(h.buttonByText("取消", modal));

  // 2) Reopen with read failure -> previous content cleared, textareas & save disabled (#58)
  mode = "fail";
  await h.click(h.buttonByText("配置训练脚本"));
  modal = h.findOne(".modal");
  assert.ok(modal.textContent.includes("读取脚本失败"));
  assert.ok(modal.textContent.includes("train.py UTF-8 解码失败"));
  const textareas2 = h.findAll("textarea", modal);
  assert.equal(textareas2[0].value, "");
  assert.equal(textareas2[0].disabled, true);
  assert.equal(textareas2[1].disabled, true);
  assert.equal(h.buttonByText("保存", modal).disabled, true);
  await h.click(h.buttonByText("取消", modal));

  // 3) Reopen cleanly, edit train.py, press Escape -> discard confirmation blocks immediate close (#71)
  mode = "ok";
  await h.click(h.buttonByText("配置训练脚本"));
  modal = h.findOne(".modal");
  const ta = h.findAll("textarea", modal)[0];
  await h.setInputValue(ta, "print('edited')");
  await h.pressKey("Escape");

  assert.ok(h.findOne(".modal"), "ScriptDialog must stay open when dirty on Escape");
  assert.ok(h.bodyText().includes("有未保存的修改，确定放弃编辑并关闭吗？"));
  await h.click(h.buttonByText("放弃修改并关闭"));
  assert.equal(h.findOne(".modal"), null);
});

test("test_t1_f4_recent_session_restart_preset_and_relay_card_drawer_toggle", async () => {
  const unverifiedRecent = {
    session_name: "llm-debug-shell",
    workspace_name: "default-ws",
    workspace_path: "D:/default-ws",
    endpoint: null,
    accelerator: "T4",
    machine_shape: "HIGH_RAM",
    created_at: "2026-11-26T10:00:00Z",
    terminated_at: "2026-11-26T11:00:00Z",
    last_event_at: "2026-11-26T11:00:00Z",
    exec_count: 1,
    file_op_count: 0,
    last_summary: "shell",
    profile_saved: false,
  };
  const savedRelayRecent = {
    session_name: "saved-relay-job",
    workspace_name: "ws-b",
    workspace_path: "D:/ws-b",
    endpoint: null,
    accelerator: "L4",
    machine_shape: "STANDARD",
    created_at: "2026-11-26T10:00:00Z",
    terminated_at: "2026-11-26T11:00:00Z",
    last_event_at: "2026-11-26T11:00:00Z",
    exec_count: 5,
    file_op_count: 2,
    last_summary: "relay",
    boot_mode: "relay",
    profile_saved: true,
    relay_config: {
      gpu_type: "L4",
      high_mem: false,
      gdrive_dir: "gdrive:custom_job",
      max_session_hours: 18,
      backup_interval_minutes: 15,
      total_epochs: 200,
      reset_done_flag: false,
    },
  };
  const runningRelayNoTelemetry = makeVm({
    name: "relay-no-telemetry",
    telemetry: null,
    relay: makeRelay("relay-no-telemetry", { running: true, phase: "provisioning", phase_label: "分配实例中" }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({
      vms: [runningRelayNoTelemetry],
      server_count: 1,
      recent_sessions: [unverifiedRecent, savedRelayRecent],
    }),
  });

  // #76: Running relay with telemetry=null must NOT show 空闲, and clicking toggle collapses & re-expands drawer
  const card = h.findOne(".vm-card");
  const topPill = h.findOne(".vm-head .pill", card);
  assert.ok(!topPill.textContent.includes("空闲"), `#76 pill must not say 空闲: ${topPill.textContent}`);
  const toggleBtn = h.findOne(".vm-tools-toggle", card);
  assert.equal(toggleBtn.getAttribute("aria-expanded"), "true");
  assert.ok(h.findOne(".vm-tools-body", card));
  await h.click(toggleBtn);
  assert.equal(toggleBtn.getAttribute("aria-expanded"), "false");
  assert.equal(h.findOne(".vm-tools-body", card), null);
  await h.click(toggleBtn);
  assert.equal(toggleBtn.getAttribute("aria-expanded"), "true");

  // #74: Restarting unverified `llm-debug-shell` opens general mode (NOT llm) with warning
  const restartBtns = h.findByText("button", "重新开机");
  await h.click(restartBtns[0]);
  let modal = h.findOne(".modal");
  assert.ok(modal.textContent.includes("该历史记录仅复用了会话名称与硬件规格"));
  assert.ok(h.buttonByText("通用算力实例 + SSH", modal).classList.contains("sel"));
  await h.click(h.buttonByText("取消", modal));

  // #74: Restarting saved relay profile restores relay mode & exact config
  await h.click(restartBtns[1]);
  modal = h.findOne(".modal");
  assert.ok(!modal.textContent.includes("该历史记录仅复用了会话名称与硬件规格"));
  assert.ok(h.buttonByText("24h+ 自动接力训练", modal).classList.contains("sel"));
  assert.ok(modal.textContent.includes("开机并启动接力训练 (L4)"));
});

// ===========================================================================
// TIER 2 — Boundary & Corner Cases (10 tests)
// ===========================================================================

test("test_t2_f1_vmcard_custom_interval_bva_30_473_600_and_rejects_29_601", async () => {
  const vm = makeVm({
    name: "ka-bva-vm",
    keep_alive: makeKeepAlive({ enabled: true, interval_seconds: 473, kernel_ping: true }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({ vms: [vm], server_count: 1 }),
    handlers: {
      set_vm_keep_alive: async ({ name, config }, ctx) => {
        const cur = ctx.getOverview();
        const nextKa = makeKeepAlive({ ...config, state: "active", profile_saved: true });
        ctx.setOverview({
          ...cur,
          vms: cur.vms.map((v) => (v.name === name ? { ...v, keep_alive: nextKa } : v)),
        });
        return nextKa;
      },
    },
  });

  const bar = h.findOne(".ka-bar");
  const input = h.findOne('input[aria-label="连接检查间隔"]', bar);
  assert.equal(input.value, "473");

  // Switch to preset 180s
  await h.click(h.buttonByText("180s", bar));
  assert.equal(h.cmds("set_vm_keep_alive").at(-1).args.config.interval_seconds, 180);

  // Switch back to custom 473s via input + Apply button
  await h.focusEl(input);
  await h.setInputValue(input, "473");
  await h.click(h.buttonByText("应用", bar));
  assert.equal(h.cmds("set_vm_keep_alive").at(-1).args.config.interval_seconds, 473);

  // Lower boundary 30s
  await h.focusEl(input);
  await h.setInputValue(input, "30");
  await h.pressKey("Enter", { target: input });
  assert.equal(h.cmds("set_vm_keep_alive").at(-1).args.config.interval_seconds, 30);

  // Upper boundary 600s
  await h.focusEl(input);
  await h.setInputValue(input, "600");
  await h.pressKey("Enter", { target: input });
  assert.equal(h.cmds("set_vm_keep_alive").at(-1).args.config.interval_seconds, 600);

  const countBeforeInvalid = h.cmds("set_vm_keep_alive").length;

  // Reject 29 (below min)
  await h.focusEl(input);
  await h.setInputValue(input, "29");
  assert.equal(input.getAttribute("aria-invalid"), "true");
  assert.ok(bar.textContent.includes("间隔须为 30~600s"));
  await h.pressKey("Enter", { target: input });
  assert.equal(h.cmds("set_vm_keep_alive").length, countBeforeInvalid);

  // Reject 601 (above max)
  await h.focusEl(input);
  await h.setInputValue(input, "601");
  assert.equal(input.getAttribute("aria-invalid"), "true");
  await h.pressKey("Enter", { target: input });
  assert.equal(h.cmds("set_vm_keep_alive").length, countBeforeInvalid);
});

test("test_t2_f1_create_vm_disabled_keepalive_bva_29_and_601_not_blocked", async () => {
  const submitted = [];
  const h = await mountApp({
    initialOverview: makeOverview(),
    handlers: {
      create_vm: async ({ req }) => {
        submitted.push(req);
        return req.name;
      },
    },
  });

  for (const invalidVal of ["29", "601"]) {
    await h.click(h.buttonByText("开机 / 新建实例"));
    const modal = h.findOne(".modal");
    const kaPanel = h.findOne(".cfg-panel-keepalive", modal);
    const kaToggle = h.findOne(".ka-inline-switch input", kaPanel);
    const kaInput = h.findOne('input[type="number"]', kaPanel);

    // Enter out-of-range value while enabled -> blocked
    await h.setInputValue(kaInput, invalidVal);
    assert.ok(kaPanel.textContent.includes("连接检查间隔必须在 30 ~ 600 秒之间"));
    assert.equal(h.buttonByText("立即开机 (T4)", modal).disabled, true);

    // Disable keep-alive -> normalized & NOT blocked (#96)
    await h.setCheckbox(kaToggle, false);
    assert.equal(kaToggle.checked, false);
    assert.equal(h.buttonByText("立即开机 (T4)", modal).disabled, false);

    await h.click(h.buttonByText("立即开机 (T4)", modal));
    assert.equal(h.findOne(".modal"), null, `CreateVmDialog should close after submit for ${invalidVal}`);
  }

  assert.equal(submitted.length, 2);
  for (const req of submitted) {
    assert.equal(req.keep_alive.enabled, false);
    assert.ok(
      req.keep_alive.interval_seconds >= 30 && req.keep_alive.interval_seconds <= 600,
      `Normalized interval must be in 30..600, got ${req.keep_alive.interval_seconds}`,
    );
  }
});

test("test_t2_f1_ping_keepalive_error_states_and_persistence_error_retry", async () => {
  let pingResponse = null;
  const vm = makeVm({
    name: "ka-err-vm",
    keep_alive: makeKeepAlive({
      enabled: false,
      interval_seconds: 120,
      kernel_ping: true,
      state: "paused",
      last_ok_at: null,
      last_attempt_at: null,
      tunnel_ok: false,
      kernel_ping_state: "disabled",
      success_count: 0,
      profile_saved: false,
      persistence_error: "磁盘只读",
    }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({ vms: [vm], server_count: 1 }),
    handlers: {
      ping_vm_keep_alive: async () => pingResponse,
      set_vm_keep_alive: async ({ config }) => makeKeepAlive({ ...config, profile_saved: true, persistence_error: null }),
    },
  });

  // 0) Decoupled paused + unsaved profile renders warn badge (not error/bad) (#97)
  const pill = h.findOne(".ka-pill");
  assert.ok(pill.className.includes("ka-pill-warn"), `Expected warn pill, got ${pill.className}`);
  assert.equal(pill.textContent.trim(), "保活未开启 · 档案未保存");
  assert.equal(h.findOne(".ka-bar-detail").textContent.trim(), "磁盘只读");

  // 0b) Ping succeeds while profile_saved=false -> warn toast (never error) (#97, #91)
  pingResponse = makeKeepAlive({
    enabled: false,
    interval_seconds: 120,
    kernel_ping: true,
    state: "paused",
    last_ok_at: 1732630500,
    last_attempt_at: 1732630500,
    tunnel_ok: true,
    kernel_ping_state: "ok",
    last_error: null,
    profile_saved: false,
    persistence_error: "磁盘只读",
  });
  await h.click(h.buttonByText("单次保活"));
  let toasts = h.toastTexts();
  const warnToast = h.findAll(".toast").find((t) => t.textContent.includes("单次检查正常") && t.textContent.includes("保活档案未保存（磁盘只读）"));
  assert.ok(warnToast, `Expected warn toast for unsaved profile on ok ping, got ${JSON.stringify(toasts)}`);
  assert.equal(h.findOne(".bar", warnToast).style.background, "var(--warning)");
  assert.ok(!toasts.some((t) => t.includes("单次检查失败")));

  // 1) Persistence error retry button (#97)
  const alertBox = h.findOne(".ka-persistence-alert");
  assert.ok(alertBox.textContent.includes("磁盘只读"));
  await h.click(h.buttonByText("重试保存策略", alertBox));
  assert.equal(h.cmds("set_vm_keep_alive").length, 1);
  assert.deepEqual(h.cmds("set_vm_keep_alive")[0].args.config, {
    enabled: false,
    interval_seconds: 120,
    kernel_ping: true,
  });

  // 2) Ping returns state="error", tunnel_ok=false, last_error="HTTP 503", profile_saved=false (#91, #97)
  pingResponse = makeKeepAlive({
    enabled: false,
    state: "error",
    tunnel_ok: false,
    kernel_ping_state: "error",
    last_error: "HTTP 503",
    profile_saved: false,
    persistence_error: "磁盘只读",
  });
  await h.click(h.buttonByText("单次保活"));
  toasts = h.toastTexts();
  const errToast = h.findAll(".toast").find((t) => t.textContent.includes("单次检查失败") && t.textContent.includes("HTTP 503"));
  assert.ok(errToast, `Expected error toast, got ${JSON.stringify(toasts)}`);
  assert.equal(h.findOne(".bar", errToast).style.background, "var(--danger)");

  // 3) Ping returns tunnel_ok=true, kernel_ping_state="error", last_error=null (#91)
  pingResponse = makeKeepAlive({
    enabled: false,
    state: "paused",
    tunnel_ok: true,
    kernel_ping_state: "error",
    last_error: null,
  });
  await h.click(h.buttonByText("单次保活"));
  toasts = h.toastTexts();
  assert.ok(toasts.some((t) => t.includes("单次检查失败") && t.includes("Kernel 心跳异常")));
});

test("test_t2_f2_max_session_hours_bva_create_and_relay_panel", async () => {
  const vm = makeVm({
    name: "relay-hours-vm",
    relay: makeRelay("relay-hours-vm", { running: false, phase: "idle", phase_label: "未开启接力" }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({ vms: [vm], server_count: 1 }),
    handlers: {
      create_vm: async ({ req }) => req.name,
      start_relay: async ({ config }) => makeRelay("relay-hours-vm", { config }),
    },
  });

  // 1) RelayPanel on VmCard BVA (#54)
  const card = h.findOne(".vm-card");
  await h.click(h.findOne(".vm-tools-toggle", card));
  const relayHoursLabel = h.findByText("label", "单棒时长 (0.1~23.5h)", card)[0];
  const relayInput = relayHoursLabel.parentNode.querySelector("input");
  const switchBtn = h.findOne(".switch", card);

  for (const invalid of ["0.05", "24", "30", "48"]) {
    await h.setInputValue(relayInput, invalid);
    assert.equal(switchBtn.disabled, true, `RelayPanel should block ${invalid}h`);
    assert.ok(card.textContent.includes("0.1 ~ 23.5"));
  }
  await h.setInputValue(relayInput, "0.1");
  assert.equal(switchBtn.disabled, false);
  await h.setInputValue(relayInput, "23.5");
  assert.equal(switchBtn.disabled, false);

  // 2) CreateVmDialog relay flow BVA (#54)
  await h.click(h.buttonByText("开机 / 新建实例"));
  const modal = h.findOne(".modal");
  await h.click(h.buttonByText("24h+ 自动接力训练", modal));
  const createHoursLabel = h.findByText("label", "单棒上限时长 (0.1~23.5h)", modal)[0];
  const createInput = createHoursLabel.parentNode.querySelector("input");
  const submitBtn = h.buttonByText("开机并启动接力训练", modal);

  for (const invalid of ["0.05", "24", "30", "48"]) {
    await h.setInputValue(createInput, invalid);
    assert.equal(submitBtn.disabled, true, `CreateVmDialog should block ${invalid}h`);
  }

  await h.setInputValue(createInput, "23.5");
  assert.equal(submitBtn.disabled, false);
  assert.ok(h.buttonByText("设为推荐 22.5h", modal));
  await h.click(h.buttonByText("设为推荐 22.5h", modal));
  assert.equal(createInput.value, "22.5");
});

test("test_t2_f2_stop_relay_dialog_destroy_and_keep_vm_boundaries", async () => {
  let stopResult = null;
  const vm = makeVm({
    name: "stop-relay-vm",
    relay: makeRelay("stop-relay-vm", { running: true }),
  });

  const h = await mountApp({
    initialOverview: makeOverview({ vms: [vm], server_count: 1 }),
    handlers: {
      stop_relay: async () => stopResult,
    },
  });

  // 1) destroy=true, verified_unassigned=false -> dialog stays open, shows remaining endpoint & NEVER success toast (#40)
  await h.click(h.findOne(".vm-card .switch.on"));
  assert.ok(h.findOne(".modal").textContent.includes("关闭 stop-relay-vm 的接力训练"));
  stopResult = {
    supervisor_stopped: true,
    destroy_vm: true,
    kernel_interrupted: false,
    kernel_interrupt_error: null,
    terminate_receipt: {
      name: "stop-relay-vm",
      endpoint: "m-s-ep1",
      verified_unassigned: false,
      remaining_endpoints: ["m-s-ep1"],
      steps: ["unassign failed"],
      ccu: null,
      error: "仍存在于计费列表",
    },
  };
  await h.click(h.buttonByText("关闭接力并彻底销毁实例"));
  let modal = h.findOne(".modal");
  assert.ok(modal, "StopRelayDialog must remain open when verified_unassigned=false");
  assert.ok(modal.textContent.includes("停费核验未通过"));
  assert.ok(modal.textContent.includes("m-s-ep1"));
  assert.ok(!h.toastTexts().some((t) => t.includes("并销毁云端实例")));
  await h.click(h.buttonByText("关闭窗口", modal));

  // 2) destroy=false, kernel_interrupted=false -> dialog stays open, warns training may continue (#41)
  await h.click(h.findOne(".vm-card .switch.on"));
  stopResult = {
    supervisor_stopped: true,
    destroy_vm: false,
    kernel_interrupted: false,
    kernel_interrupt_error: "restart-kernel timeout",
    terminate_receipt: null,
  };
  await h.click(h.buttonByText("仅关闭接力，保留实例"));
  modal = h.findOne(".modal");
  assert.ok(modal, "StopRelayDialog must remain open when kernel_interrupted=false");
  assert.ok(modal.textContent.includes("未能确认中断"));
  assert.ok(modal.textContent.includes("restart-kernel timeout"));
  assert.ok(h.buttonByText("直接销毁实例并停费", modal));

  // 3) destroy=true, verified_unassigned=true -> closes cleanly with success toast (#40)
  stopResult = {
    supervisor_stopped: true,
    destroy_vm: true,
    kernel_interrupted: true,
    kernel_interrupt_error: null,
    terminate_receipt: {
      name: "stop-relay-vm",
      endpoint: "m-s-ep1",
      verified_unassigned: true,
      remaining_endpoints: [],
      steps: ["ok"],
      ccu: makeCcu({ burn_rate_hourly: 0, assignments_count: 0 }),
      error: null,
    },
  };
  await h.click(h.buttonByText("直接销毁实例并停费", modal));
  assert.equal(h.findOne(".modal"), null);
  assert.ok(h.toastTexts().some((t) => t.includes("已关闭 [stop-relay-vm] 接力并销毁云端实例")));
});

test("test_t2_f3_chat_modal_blocks_warming_and_resets_vanished_target", async () => {
  const readyVm = makeVm({
    name: "llm-blue-ready",
    llm_api: makeLlmApi("llm-blue-ready", { slot: "blue" }),
    telemetry: {
      ...makeVm().telemetry,
      llm: { online: true, backend_healthy: true, slot: "blue", role: "active", model_name: "Qwen-7B", instant_tps: 0 },
    },
  });
  const warmingVm = makeVm({
    name: "llm-green-warming",
    llm_api: makeLlmApi("llm-green-warming", { slot: "green" }),
    telemetry: {
      ...makeVm().telemetry,
      llm: { online: false, backend_healthy: false, slot: "green", role: "warming", model_name: "Qwen-7B", instant_tps: 0 },
    },
  });
  const secondReadyVm = makeVm({
    name: "llm-green-ready",
    llm_api: makeLlmApi("llm-green-ready", { slot: "green" }),
    telemetry: {
      ...makeVm().telemetry,
      llm: { online: true, backend_healthy: true, slot: "green", role: "standby", model_name: "Qwen-7B", instant_tps: 0 },
    },
  });

  const h = await mountApp({
    initialOverview: makeOverview({
      vms: [readyVm, warmingVm, secondReadyVm],
      server_count: 3,
      llm_bluegreen: { active_slot: "blue", active_session: "llm-blue-ready", blue_session: "llm-blue-ready" },
    }),
    handlers: {
      stream_llm_chat: async ({ req }) => ({
        request_id: req.request_id,
        vm_name: req.target_vm,
        slot: "blue",
        model: "Qwen-7B",
        reply: "ok",
        ttft_ms: 50,
        avg_tps: 20,
        completion_tokens: 4,
        elapsed_ms: 200,
        transport: "gateway",
      }),
    },
  });

  // Open chat targeting warming node directly -> send blocked (#64)
  await h.click(h.buttonByText("大模型对话"));
  const modal = h.findOne(".chat-modal");
  const select = h.findOne("select.chat-target-select", modal);
  const warmingOpt = h.findAll("option", select).find((o) => o.value === "llm-green-warming");
  assert.equal(warmingOpt.disabled, true, "#64 warming option must be disabled");

  // Select secondReadyVm, then emit overview removing secondReadyVm -> resets to auto + notice (#64)
  await h.selectOption(select, "llm-green-ready");
  assert.equal(select.value, "llm-green-ready");

  await h.emitEvent(
    "overview",
    makeOverview({
      seq: 2,
      timestamp: 1732630010,
      vms: [readyVm, warmingVm],
      server_count: 2,
      llm_bluegreen: { active_slot: "blue", active_session: "llm-blue-ready", blue_session: "llm-blue-ready" },
    }),
  );

  assert.equal(select.value, "auto", "#64 vanished target must reset select to auto");
  assert.ok(modal.textContent.includes("原选定节点 [llm-green-ready] 已离线或被释放"));

  const ta = h.findOne("textarea.chat-textarea", modal);
  await h.setInputValue(ta, "hello");
  await h.click(h.buttonByText("发送", modal));
  assert.equal(h.cmds("stream_llm_chat").length, 1);
  assert.equal(h.cmds("stream_llm_chat")[0].args.req.target_vm, "llm-blue-ready");
});

test("test_t2_f4_modal_scroll_and_compact_viewport_css_ast", async () => {
  // 1) PostCSS AST verification on src/styles.css (#66)
  const modalDecls = STYLES_AST.getMergedDecls(".modal");
  assert.equal(modalDecls["flex-direction"], "column");
  assert.equal(modalDecls["min-height"], "0");
  assert.equal(modalDecls["overflow"], "hidden");

  const bodyDecls = STYLES_AST.getMergedDecls(".modal-body");
  assert.ok(bodyDecls["flex"].startsWith("1 1 auto"));
  assert.equal(bodyDecls["min-height"], "0");
  assert.equal(bodyDecls["overflow-y"], "auto");

  const bodyChildDecls = STYLES_AST.getMergedDecls(".modal-body > *");
  assert.equal(bodyChildDecls["flex-shrink"], "0");

  const advDecls = STYLES_AST.getMergedDecls(".adv-config-box");
  assert.equal(advDecls["flex-shrink"], "0");
  assert.notEqual(advDecls["overflow"], "hidden");

  const compactModal = STYLES_AST.getMergedDecls(".modal", "max-height: 640px");
  assert.ok(compactModal["max-height"], "@media (max-height: 640px) must adjust .modal max-height");
  const compactHead = STYLES_AST.getMergedDecls(".modal-head", "max-height: 640px");
  const compactFoot = STYLES_AST.getMergedDecls(".modal-foot", "max-height: 640px");
  assert.ok(compactHead.padding && compactFoot.padding);

  // 2) Live DOM 3-part structure check on CreateVmDialog with LLM + advanced expanded
  const h = await mountApp({ initialOverview: makeOverview() });
  await h.click(h.buttonByText("开机 / 新建实例"));
  const modal = h.findOne(".modal");
  assert.ok(h.findOne(".modal-head", modal));
  const modalBody = h.findOne(".modal-body", modal);
  assert.ok(modalBody);
  assert.ok(h.findOne(".modal-foot", modal));
  await h.click(h.buttonByText("部署大模型 + API / Key", modal));
  await h.click(h.buttonByText("更多 Colab 运行时与环境配置", modal));
  assert.ok(h.findOne(".adv-config-body", modalBody));
});

test("test_t2_f4_modal_tab_shift_tab_focus_trap_and_restore", async () => {
  const h = await mountApp({
    initialOverview: makeOverview(),
    handlers: {
      read_workspace_files: async () => ({
        workspace: "D:/default-ws",
        train_py: "print(1)",
        requirements_txt: "torch\n",
      }),
    },
  });

  const triggerBtn = h.buttonByText("配置训练脚本");
  await h.focusEl(triggerBtn);
  assert.equal(h.doc.activeElement, triggerBtn);

  await h.click(triggerBtn);
  const modal = h.findOne(".modal");
  await h.flush(5);
  assert.ok(modal.contains(h.doc.activeElement), "#70 initial focus must move inside .modal");

  // Find focusables inside modal
  const focusables = h
    .findAll("button, textarea, input, select", modal)
    .filter((el) => !el.disabled);
  assert.ok(focusables.length >= 3);

  // Move to last focusable, press Tab -> wraps to first
  await h.focusEl(focusables[focusables.length - 1]);
  await h.pressKey("Tab");
  assert.equal(h.doc.activeElement, focusables[0], "#70 Tab at end must wrap to first focusable");

  // At first focusable, press Shift+Tab -> wraps to last
  await h.pressKey("Tab", { shiftKey: true });
  assert.equal(
    h.doc.activeElement,
    focusables[focusables.length - 1],
    "#70 Shift+Tab at start must wrap to last focusable",
  );

  // Background focus attempt must be pulled back inside modal
  const bgBtn = h.buttonByText("开机 / 新建实例");
  await h.focusEl(bgBtn);
  assert.ok(modal.contains(h.doc.activeElement), "#70 background focus must be trapped back into modal");

  // Close modal -> focus restores to trigger button
  await h.click(h.buttonByText("取消", modal));
  assert.equal(h.findOne(".modal"), null);
  assert.equal(h.doc.activeElement, triggerBtn, "#70 focus must restore to trigger button on close");
});

test("test_t2_f4_unsaved_script_and_relay_close_stacking_over_chat", async () => {
  // Check z-index order in styles.css AST (#87)
  const overlayZ = Number(STYLES_AST.getMergedDecls(".overlay")["z-index"]);
  const topmostZ = Number(STYLES_AST.getMergedDecls(".overlay.overlay-topmost")["z-index"]);
  const toastsZ = Number(STYLES_AST.getMergedDecls(".toasts")["z-index"]);
  assert.ok(overlayZ < topmostZ && topmostZ < toastsZ, `Expected ${overlayZ} < ${topmostZ} < ${toastsZ}`);

  const readyLlm = makeVm({
    name: "llm-blue-stack",
    llm_api: makeLlmApi("llm-blue-stack"),
    telemetry: {
      ...makeVm().telemetry,
      llm: { online: true, backend_healthy: true, slot: "blue", role: "active", model_name: "Qwen-7B", instant_tps: 0 },
    },
  });

  const h = await mountApp({
    initialOverview: makeOverview({ vms: [readyLlm], server_count: 1 }),
    handlers: {
      read_workspace_files: async () => ({ workspace: "D:/default-ws", train_py: "v1", requirements_txt: "" }),
      confirm_app_close: async () => [],
    },
  });

  // Open chat modal
  await h.click(h.buttonByText("大模型对话"));
  assert.ok(h.findOne(".chat-modal"));

  // Emit close-requested-unsaved-script while chat modal is open -> CloseAppUnsavedScriptDialog stacks on top (#83, #87)
  await h.emitEvent("close-requested-unsaved-script", { active_vms: ["relay-1"] });
  const overlays = h.findAll(".overlay");
  assert.equal(overlays.length, 2);
  assert.equal(overlays[0].querySelector(".chat-modal") !== null, true);
  assert.ok(overlays[1].classList.contains("overlay-topmost"));
  assert.ok(overlays[1].textContent.includes("有未保存的脚本修改，确认关闭主窗口？"));

  // Escape closes ONLY the topmost close dialog, leaving chat modal open
  await h.pressKey("Escape");
  assert.equal(h.findAll(".overlay").length, 1);
  assert.ok(h.findOne(".chat-modal"), "Escape on topmost dialog must preserve chat modal beneath");

  // Re-emit and click 放弃修改并关闭 -> transitions to CloseAppRelayDialog (#72, #83, #87)
  await h.emitEvent("close-requested-unsaved-script", { active_vms: ["relay-1"] });
  await h.click(h.buttonByText("放弃修改并关闭"));
  assert.ok(h.findOne(".overlay.overlay-topmost").textContent.includes("检测到正在运行的自动接力训练任务"));
});

test("test_t2_f4_script_dirty_ipc_transitions_on_edit_save_and_discard", async () => {
  let savedPayload = null;
  const h = await mountApp({
    initialOverview: makeOverview(),
    handlers: {
      read_workspace_files: async () => ({ workspace: "D:/default-ws", train_py: "orig", requirements_txt: "req1" }),
      save_workspace_files: async (args) => {
        savedPayload = args;
        return null;
      },
    },
  });

  await h.click(h.buttonByText("配置训练脚本"));
  let modal = h.findOne(".modal");
  let ta = h.findAll("textarea", modal)[0];

  // Edit -> set_script_dirty(true)
  await h.setInputValue(ta, "modified");
  assert.equal(h.cmds("set_script_dirty").at(-1).args.dirty, true);

  // Revert back to orig -> set_script_dirty(false)
  await h.setInputValue(ta, "orig");
  assert.equal(h.cmds("set_script_dirty").at(-1).args.dirty, false);

  // Edit again and save -> save_workspace_files + set_script_dirty(false)
  await h.setInputValue(ta, "saved_v2");
  assert.equal(h.cmds("set_script_dirty").at(-1).args.dirty, true);
  await h.click(h.buttonByText("保存", modal));
  assert.equal(savedPayload.trainPy, "saved_v2");
  assert.equal(h.cmds("set_script_dirty").at(-1).args.dirty, false);

  // Reopen, edit, Escape -> 放弃修改并关闭 -> set_script_dirty(false)
  await h.click(h.buttonByText("配置训练脚本"));
  modal = h.findOne(".modal");
  ta = h.findAll("textarea", modal)[0];
  await h.setInputValue(ta, "discard_me");
  assert.equal(h.cmds("set_script_dirty").at(-1).args.dirty, true);
  await h.pressKey("Escape");
  await h.click(h.buttonByText("放弃修改并关闭"));
  assert.equal(h.cmds("set_script_dirty").at(-1).args.dirty, false);
});

// ===========================================================================
// TIER 3 — Cross-Feature Pairwise Combinations (4 tests)
// ===========================================================================

test("test_t3_f1_f4_create_vm_keepalive_disabled_with_env_and_auth_form", async () => {
  let createdReq = null;
  const h = await mountApp({
    initialOverview: makeOverview({
      env: makeEnv({ token_ok: false, adc_ok: true, rclone_conf_ok: false }),
    }),
    handlers: {
      create_vm: async ({ req }) => {
        createdReq = req;
        return req.name;
      },
    },
  });

  await h.click(h.buttonByText("开机 / 新建实例"));
  const modal = h.findOne(".modal");

  // Pairwise F1 (disabled keepalive after invalid 601) x F4 (ADC block -> OAuth2 + general rclone optout + multiline env)
  const kaPanel = h.findOne(".cfg-panel-keepalive", modal);
  const kaInput = h.findOne('input[type="number"]', kaPanel);
  const kaToggle = h.findOne(".ka-inline-switch input", kaPanel);
  await h.setInputValue(kaInput, "601");
  await h.setCheckbox(kaToggle, false);

  await h.click(h.buttonByText("ADC 凭证", modal));
  assert.equal(h.buttonByText("立即开机 (T4)", modal).disabled, true);
  await h.click(h.buttonByText("OAuth2 账号", modal));

  await h.click(h.buttonByText("更多 Colab 运行时与环境配置", modal));
  const envTa = h.findOne('textarea[placeholder*="HF_TOKEN=hf_xxx"]', modal);
  await h.setInputValue(envTa, "A=1\nB=2");

  await h.click(h.buttonByText("立即开机 (T4)", modal));
  assert.ok(createdReq);
  assert.equal(createdReq.keep_alive.enabled, false);
  assert.ok(createdReq.keep_alive.interval_seconds >= 30 && createdReq.keep_alive.interval_seconds <= 600);
  assert.equal(createdReq.env_config.sync_rclone, false);
  assert.equal(createdReq.env_config.env_vars, "A=1\nB=2");
});

test("test_t3_f2_f4_relay_active_close_guard_over_chat_and_stale_billing", async () => {
  const activeRelay = makeVm({
    name: "relay-active-1",
    relay: makeRelay("relay-active-1", { running: true }),
  });
  const readyLlm = makeVm({
    name: "llm-blue-1",
    llm_api: makeLlmApi("llm-blue-1"),
    telemetry: {
      ...makeVm().telemetry,
      llm: { online: true, backend_healthy: true, slot: "blue", role: "active", model_name: "Qwen-7B", instant_tps: 0 },
    },
  });

  let closeMode = null;
  const h = await mountApp({
    initialOverview: makeOverview({
      vms: [activeRelay, readyLlm],
      server_count: 2,
      llm_bluegreen: { active_slot: "blue", active_session: "llm-blue-1", blue_session: "llm-blue-1" },
    }),
    handlers: {
      confirm_app_close: async ({ mode }) => {
        closeMode = mode;
        return [
          {
            name: "relay-active-1",
            endpoint: "m-s-ep1",
            verified_unassigned: false,
            remaining_endpoints: ["m-s-ep1"],
            steps: ["unassign timeout"],
            ccu: null,
            error: "仍计费",
          },
        ];
      },
    },
  });

  // Open chat modal, then emit server_list_known=false + close-requested-with-relay
  await h.click(h.buttonByText("大模型对话"));
  await h.emitEvent(
    "overview",
    makeOverview({
      seq: 2,
      timestamp: 1732630050,
      server_list_known: false,
      ccu: null,
      vms: [{ ...activeRelay, on_server: false, server_state_unknown: true, telemetry: null }, readyLlm],
    }),
  );
  await h.emitEvent("close-requested-with-relay", { active_vms: ["relay-active-1"] });

  const topmost = h.findOne(".overlay.overlay-topmost");
  assert.ok(topmost.textContent.includes("relay-active-1"));
  assert.ok(topmost.textContent.includes("将继续产生计费"));

  // Choose shutdown_vms which fails verification -> dialog remains open with unknown burn rate (never 0.00 CU/h)
  await h.click(h.buttonByText("停止接力并关机停费后退出", topmost));
  assert.equal(closeMode, "shutdown_vms");
  assert.ok(h.findOne(".overlay.overlay-topmost").textContent.includes("仍计费"));
  assert.ok(h.findOne(".overlay.overlay-topmost").textContent.includes("未知"));
  assert.ok(!h.findOne(".overlay.overlay-topmost").textContent.includes("0.00 CU/h"));
});

test("test_t3_f3_f4_bluegreen_chat_modal_focus_trap_and_tps_snapshot_order", async () => {
  const customBlue = makeVm({
    name: "prod-deepseek-node",
    llm_api: makeLlmApi("prod-deepseek-node", { slot: "blue" }),
    telemetry: {
      ...makeVm().telemetry,
      llm: {
        online: true,
        backend_healthy: true,
        slot: "blue",
        role: "active",
        model_name: "DeepSeek-R1-7B",
        instant_tps: 0,
        last_request_tps: 41.5,
      },
    },
  });

  const h = await mountApp({
    initialOverview: makeOverview({
      seq: 10,
      timestamp: 3000,
      last_ok_timestamp: 3000,
      vms: [customBlue],
      server_count: 1,
      llm_bluegreen: {
        active_slot: "blue",
        active_session: "prod-deepseek-node",
        blue_session: "prod-deepseek-node",
        model_name: "DeepSeek-R1-7B",
      },
    }),
    handlers: {
      stream_llm_chat: async ({ req }) => ({
        request_id: req.request_id,
        vm_name: req.target_vm,
        slot: "blue",
        model: "DeepSeek-R1-7B",
        reply: "pong",
        ttft_ms: 80,
        avg_tps: 45,
        completion_tokens: 5,
        elapsed_ms: 110,
        transport: "gateway",
      }),
    },
  });

  // Open chat modal and verify focus trap + custom blue session routing (#48, #65, #70)
  await h.click(h.buttonByText("大模型对话"));
  const modal = h.findOne(".chat-modal");
  assert.ok(modal.textContent.includes("0.0 tok/s (空闲)"));
  assert.ok(modal.textContent.includes("上次请求: 41.5 tok/s"));
  assert.ok(modal.textContent.includes("prod-deepseek-node"));

  // Late older snapshot seq=9 must be ignored (#82)
  await h.emitEvent(
    "overview",
    makeOverview({
      seq: 9,
      timestamp: 2900,
      last_ok_timestamp: 2900,
      is_stale: true,
      vms: [],
    }),
  );
  assert.ok(modal.textContent.includes("prod-deepseek-node"));

  // Focus trap check inside chat modal (#70)
  const bgBtn = h.buttonByText("开机 / 新建实例");
  await h.focusEl(bgBtn);
  assert.ok(modal.contains(h.doc.activeElement));

  // Send message -> targets custom-named prod-deepseek-node (#48)
  const ta = h.findOne("textarea.chat-textarea", modal);
  await h.setInputValue(ta, "ping");
  await h.click(h.buttonByText("发送", modal));
  assert.equal(h.cmds("stream_llm_chat")[0].args.req.target_vm, "prod-deepseek-node");
});

test("test_t3_f2_f3_dashboard_mixed_relay_and_bluegreen_cards_lifecycle", async () => {
  const relayVm = makeVm({
    name: "relay-mix",
    workspace: { name: "ws-mix", path: "D:/ws-mix", remote_cwd: "/content/ws-mix", source: "custom", active_script: "train.py", description: null },
    telemetry: null,
    relay: makeRelay("relay-mix", {
      running: false,
      phase: "download_failed",
      training_completed: true,
      weights_downloaded: false,
      download_error: "rclone timeout",
      last_error: "检测到旧版默认目录 gdrive:colab_relay 存在历史断点",
    }),
  });
  const llmVm = makeVm({
    name: "custom-llm-active",
    llm_api: makeLlmApi("custom-llm-active"),
    telemetry: {
      ...makeVm().telemetry,
      llm: {
        online: true,
        backend_healthy: true,
        slot: "blue",
        role: "active",
        model_name: "Qwen-7B",
        instant_tps: 0,
        last_request_tps: 33.2,
      },
    },
  });

  let startArgs = null;
  const h = await mountApp({
    initialOverview: makeOverview({
      vms: [relayVm, llmVm],
      server_count: 2,
      llm_bluegreen: { active_slot: "blue", active_session: "custom-llm-active", blue_session: "custom-llm-active" },
    }),
    handlers: {
      start_relay: async (args) => {
        startArgs = args;
        return makeRelay("relay-mix", { running: true });
      },
    },
  });

  // Verify both cards coexist cleanly
  const cards = h.findAll(".vm-card");
  assert.equal(cards.length, 2);

  // Relay card shows download failed (#53) + legacy migration prompt (#80) + drawer toggle (#76)
  const cardRelay = cards[0];
  await h.click(h.findOne(".vm-tools-toggle", cardRelay));
  assert.ok(cardRelay.textContent.includes("云端训练已完成，但最终权重下载失败"));
  assert.ok(cardRelay.textContent.includes("检测到旧版根目录存在历史断点"));
  await h.click(h.buttonByText("迁移旧断点并续训", cardRelay));
  assert.equal(startArgs.workspacePath, "D:/ws-mix", "#42 workspace routing");
  assert.equal(startArgs.legacyGdriveAction, "migrate", "#80 migration action");

  // Bluegreen banner + LLM card show custom-llm-active & idle tps (#48, #65)
  assert.ok(h.findOne(".bg-cluster-card").textContent.includes("custom-llm-active"));
  assert.ok(cards[1].textContent.includes("0.0 tok/s (空闲)"));
  assert.ok(cards[1].textContent.includes("上次请求: 33.2 tok/s"));
});

// ===========================================================================
// TIER 4 — Real-World Application Scenarios (2 tests)
// ===========================================================================

test("test_t4_w5_full_ui_dl_relay_create_edit_stop_and_recovery_workflow", async () => {
  let stepOverview = makeOverview({
    env: makeEnv({ rclone_conf_ok: true }),
    vms: [],
    server_count: 0,
  });

  const h = await mountApp({
    initialOverview: stepOverview,
    handlers: {
      refresh_now: async () => stepOverview,
      create_vm: async ({ req }) => {
        // Return persistence_error on created VM (#97) + disabled keepalive normalized (#96) + CPU (#52)
        stepOverview = makeOverview({
          seq: 2,
          timestamp: 1732630010,
          vms: [
            makeVm({
              name: req.name,
              accelerator: req.accelerator,
              high_ram: req.high_mem,
              workspace: { name: "ws-w5", path: req.workspace_path, remote_cwd: "/content/ws-w5", source: "custom", active_script: "train.py", description: null },
              keep_alive: makeKeepAlive({
                enabled: true,
                interval_seconds: 120,
                profile_saved: false,
                persistence_error: "profiles write failed",
              }),
              telemetry: null,
              relay: makeRelay(req.name, { running: true, config: req.relay_config }),
            }),
          ],
          server_count: 1,
        });
        return req.name;
      },
      set_vm_keep_alive: async ({ name, config }) => {
        const ka = makeKeepAlive({ ...config, profile_saved: true, persistence_error: null });
        stepOverview = makeOverview({
          ...stepOverview,
          seq: 3,
          timestamp: 1732630020,
          vms: stepOverview.vms.map((v) => (v.name === name ? { ...v, keep_alive: ka } : v)),
        });
        return ka;
      },
      read_workspace_files: async ({ workspacePath }) => ({
        workspace: workspacePath,
        train_py: "print('round 1')",
        requirements_txt: "torch\n",
      }),
      save_workspace_files: async () => null,
      stop_relay: async (_args) => ({
        supervisor_stopped: true,
        destroy_vm: false,
        kernel_interrupted: false,
        kernel_interrupt_error: "timeout",
        terminate_receipt: null,
      }),
      start_relay: async ({ name, config }) => makeRelay(name, { running: true, config }),
      dismiss_relay_status: async () => {
        stepOverview = makeOverview({ ...stepOverview, seq: 6, timestamp: 1732630050, vms: [] });
      },
    },
  });

  // 1. Create CPU Relay VM with custom workspace, disabled invalid keepalive 601 (#96), max_session_hours BVA (#54), isolated dir (#44)
  await h.click(h.buttonByText("开机 / 新建实例"));
  let modal = h.findOne(".modal");
  await h.click(h.buttonByText("24h+ 自动接力训练", modal));
  await h.click(h.hwButton("CPU", modal));
  const nameInput = h.findOne('input[placeholder="例如 gpu-dev-01"]', modal);
  await h.setInputValue(nameInput, "w5-cpu-relay");
  assert.ok(modal.textContent.includes("gdrive:colab_relay/w5-cpu-relay"));

  const kaPanel = h.findOne(".cfg-panel-keepalive", modal);
  await h.setInputValue(h.findOne('input[type="number"]', kaPanel), "601");
  await h.setCheckbox(h.findOne(".ka-inline-switch input", kaPanel), false);

  await h.click(h.buttonByText("更多 Colab 运行时与环境配置", modal));
  await h.setInputValue(h.findOne('input[placeholder="默认绑定当前项目工作区目录"]', modal), "D:/ws-w5");
  await h.click(h.buttonByText("开机并启动接力训练 (CPU)", modal));

  assert.equal(h.cmds("create_vm")[0].args.req.relay_config.gpu_type, "CPU");
  assert.equal(h.cmds("create_vm")[0].args.req.keep_alive.enabled, false);

  // 2. Persistence error surfaced on VmCard (#97) -> retry + custom interval 473 (#98)
  const card = h.findOne(".vm-card");
  assert.ok(card.textContent.includes("profiles write failed"));
  await h.click(h.buttonByText("重试保存策略", card));
  const kaInput = h.findOne('input[aria-label="连接检查间隔"]', card);
  await h.focusEl(kaInput);
  await h.setInputValue(kaInput, "473");
  await h.pressKey("Enter", { target: kaInput });
  assert.equal(h.cmds("set_vm_keep_alive").at(-1).args.config.interval_seconds, 473);

  // 3. Toggle drawer (#76) & edit workspace script with dirty Esc guard (#42, #58, #71)
  await h.click(h.findOne(".vm-tools-toggle", card));
  assert.equal(h.findOne(".vm-tools-body", card), null);
  await h.click(h.findOne(".vm-tools-toggle", card));
  await h.click(h.buttonByText("编辑训练脚本", card));
  modal = h.findOne(".modal");
  await h.setInputValue(h.findAll("textarea", modal)[0], "print('round 2')");
  await h.pressKey("Escape");
  assert.ok(h.bodyText().includes("有未保存的修改"));
  await h.click(h.buttonByText("继续编辑", modal));
  await h.click(h.buttonByText("保存", modal));
  assert.equal(h.cmds("save_workspace_files")[0].args.workspacePath, "D:/ws-w5");

  // 4. Stop relay keep-vm boundary failure (#41), then download failed retry (#53), legacy migration (#80), offline dismiss (#73)
  await h.click(h.findOne(".vm-card .switch.on"));
  await h.click(h.buttonByText("仅关闭接力，保留实例"));
  assert.ok(h.findOne(".modal").textContent.includes("未能确认中断"));
  await h.click(h.buttonByText("关闭窗口"));

  await h.emitEvent(
    "overview",
    makeOverview({
      seq: 4,
      timestamp: 1732630030,
      vms: [
        makeVm({
          name: "w5-cpu-relay",
          on_server: false,
          telemetry: null,
          relay: makeRelay("w5-cpu-relay", {
            running: false,
            phase: "download_failed",
            training_completed: true,
            weights_downloaded: false,
            download_error: "disk full",
          }),
        }),
      ],
    }),
  );
  await h.click(h.buttonByText("重试补下载权重"));
  assert.equal(h.cmds("start_relay").at(-1).args.config.reset_done_flag, false);

  await h.emitEvent(
    "overview",
    makeOverview({
      seq: 5,
      timestamp: 1732630040,
      vms: [
        makeVm({
          name: "w5-cpu-relay",
          on_server: false,
          telemetry: null,
          relay: makeRelay("w5-cpu-relay", {
            running: false,
            phase: "error",
            last_error: "检测到旧版默认目录 gdrive:colab_relay 存在历史断点",
          }),
        }),
      ],
    }),
  );
  await h.click(h.buttonByText("迁移旧断点并续训"));
  assert.equal(h.cmds("start_relay").at(-1).args.legacyGdriveAction, "migrate");

  await h.click(h.buttonByText("清除记录"));
  assert.equal(h.cmds("dismiss_relay_status").length, 1);
  assert.equal(h.findAll(".vm-card").length, 0);
});

test("test_t4_w6_full_ui_bluegreen_chat_billing_and_close_guard_workflow", async () => {
  const customBlue = makeVm({
    name: "prod-qwen-node",
    keep_alive: makeKeepAlive({ enabled: false, state: "paused", tunnel_ok: false, kernel_ping_state: "disabled" }),
    llm_api: makeLlmApi("prod-qwen-node", { slot: "blue" }),
    telemetry: {
      ...makeVm().telemetry,
      llm: {
        online: true,
        backend_healthy: true,
        slot: "blue",
        role: "active",
        model_name: "Qwen/Qwen2.5-7B-Instruct-GGUF",
        instant_tps: 0,
        last_request_tps: 38,
      },
    },
  });
  const warmingGreen = makeVm({
    name: "prod-qwen-green",
    llm_api: makeLlmApi("prod-qwen-green", { slot: "green" }),
    telemetry: {
      ...makeVm().telemetry,
      llm: { online: false, backend_healthy: false, slot: "green", role: "warming", model_name: "Qwen/Qwen2.5-7B-Instruct-GGUF", instant_tps: 0 },
    },
  });

  let latestOv = makeOverview({
    seq: 10,
    timestamp: 5000,
    last_ok_timestamp: 5000,
    ccu: null,
    server_count: 2,
    vms: [customBlue, warmingGreen],
    llm_bluegreen: {
      active_slot: "blue",
      active_session: "prod-qwen-node",
      blue_session: "prod-qwen-node",
      green_session: "prod-qwen-green",
      model_name: "Qwen/Qwen2.5-7B-Instruct-GGUF",
    },
    recent_sessions: [
      {
        session_name: "llm-unverified",
        workspace_name: "default-ws",
        workspace_path: "D:/default-ws",
        endpoint: null,
        accelerator: "T4",
        machine_shape: "HIGH_RAM",
        created_at: null,
        terminated_at: null,
        last_event_at: "2026-11-26T10:00:00Z",
        exec_count: 1,
        file_op_count: 0,
        last_summary: "test",
        profile_saved: false,
      },
    ],
  });

  const h = await mountApp({
    initialOverview: latestOv,
    handlers: {
      refresh_now: async () => latestOv,
      ping_vm_keep_alive: async () =>
        makeKeepAlive({
          enabled: false,
          state: "paused",
          tunnel_ok: false,
          kernel_ping_state: "error",
          last_error: "503",
        }),
      rotate_llm_api_key: async () => {
        throw new Error("disk locked");
      },
      stream_llm_chat: async ({ req }) => ({
        request_id: req.request_id,
        vm_name: req.target_vm,
        slot: "blue",
        model: "Qwen/Qwen2.5-7B-Instruct-GGUF",
        reply: "streamed reply",
        ttft_ms: 95,
        avg_tps: 39.2,
        completion_tokens: 12,
        elapsed_ms: 300,
        transport: "gateway",
      }),
      read_workspace_files: async () => ({ workspace: "D:/default-ws", train_py: "x=1", requirements_txt: "" }),
      confirm_app_close: async () => [],
    },
  });

  // 1. Missing CCU shows 未知 (never 0.00 CU/h) (#56) + idle TPS 0.0 tok/s (空闲) & 上次请求: 38.0 tok/s (#65)
  assert.ok(h.findOne("header.topbar").textContent.includes("未知"));
  assert.ok(!h.findOne("header.topbar").textContent.includes("0.00 CU/h"));
  assert.ok(h.bodyText().includes("0.0 tok/s (空闲)"));
  assert.ok(h.bodyText().includes("上次请求: 38.0 tok/s"));

  // 2. Paused VM single ping failure surfaces error toast & never enables periodic policy (#91)
  await h.click(h.buttonByText("单次保活", h.findAll(".vm-card")[0]));
  assert.equal(h.cmds("set_vm_keep_alive").length, 0);
  assert.ok(h.toastTexts().some((t) => t.includes("单次检查失败") && t.includes("503")));

  // 3. Key rotation error (#59)
  await h.click(h.findOne('button[title="重新生成此虚拟机的 API Key"]', h.findAll(".vm-card")[0]));
  assert.ok(h.toastTexts().some((t) => t.includes("轮换 API Key 失败") && t.includes("disk locked")));

  // 4. Stale manual refresh (#57) + older snapshot rejected (#82)
  latestOv = makeOverview({ ...latestOv, seq: 11, timestamp: 5060, last_ok_timestamp: 4950, is_stale: true, stale_reason: "timeout", vms: [] });
  await h.click(h.findOne('header.topbar button[title*="立即刷新"]'));
  assert.ok(h.bodyText().includes("最新查询失败：timeout"));
  assert.ok(h.bodyText().includes("prod-qwen-node"));

  // 5. Recent unverified `llm-unverified` -> general mode (#74) + CUDA CPU guard (#68)
  await h.click(h.buttonByText("重新开机"));
  let modal = h.findOne(".modal");
  assert.ok(h.buttonByText("通用算力实例 + SSH", modal).classList.contains("sel"));
  await h.click(h.buttonByText("部署大模型 + API / Key", modal));
  await h.click(h.hwButton("CPU", modal));
  assert.equal(h.buttonByText("开机并部署大模型 (CPU)", modal).disabled, true);
  await h.click(h.buttonByText("取消", modal));

  // 6. Open chat modal -> warming disabled (#64), focus trap (#70), sends to custom prod-qwen-node (#48)
  await h.click(h.buttonByText("大模型对话"));
  const chatModal = h.findOne(".chat-modal");
  const select = h.findOne("select.chat-target-select", chatModal);
  assert.equal(h.findAll("option", select).find((o) => o.value === "prod-qwen-green").disabled, true);
  await h.setInputValue(h.findOne("textarea.chat-textarea", chatModal), "hi");
  await h.click(h.buttonByText("发送", chatModal));
  assert.equal(h.cmds("stream_llm_chat")[0].args.req.target_vm, "prod-qwen-node");

  // 7. Unsaved script close guard stacks above chat modal (#83, #87)
  await h.emitEvent("close-requested-unsaved-script", { active_vms: ["relay-x"] });
  const topmost = h.findOne(".overlay.overlay-topmost");
  assert.ok(topmost.textContent.includes("有未保存的脚本修改"));
  await h.click(h.buttonByText("放弃修改并关闭", topmost));
  assert.ok(h.findOne(".overlay.overlay-topmost").textContent.includes("relay-x"));
});
