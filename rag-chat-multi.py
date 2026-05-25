#!/usr/bin/env python3
"""
rag-chat-multi.py — corrected full file with fixed iframe sizing (prevents 200px collapse)

Notes:
 - Parent CSS now sets an explicit iframe height using calc(100vh - composer - padding) !important
 - We also supply a large fallback height to components.html to avoid early collapse
 - The chat pane scrolls internally (.rag-messages overflow-y:auto)
 - Composer still pinned to bottom center; uploader auto-clears by rotating key
"""

from __future__ import annotations

import html as html_lib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# ---------------- Page config ----------------
st.set_page_config(page_title="RAG Chat — Single Conversation", layout="wide", initial_sidebar_state="collapsed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger("rag_chat_multi_iframe_fix")

# ---------------- Constants ----------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CENTER_WIDTH_PX = 760
COMPOSER_HEIGHT = 72  # px reserved for the pinned composer
IFRAME_FALLBACK_HEIGHT = 600  # large fallback so parent CSS can override reliably

# ---------------- OpenAI client ----------------
client = OpenAI()

# ---------------- Helpers ----------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def build_attachment_preview_list(files) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not files:
        return out
    for f in files:
        try:
            name = getattr(f, "name", None) or (f.name if hasattr(f, "name") else str(f))
        except Exception:
            name = str(f)
        out.append({"name": name})
    return out

# ---------------- Session defaults ----------------
if "conversation" not in st.session_state:
    st.session_state.conversation = {"id": str(uuid.uuid4()), "messages": []}
if "composer_text" not in st.session_state:
    st.session_state.composer_text = ""
if "composer_api_key" not in st.session_state:
    st.session_state.composer_api_key = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

# ---------------- Model helper ----------------
def list_models_for_select(max_items: int = 40) -> List[str]:
    try:
        resp = client.models.list()
        models: List[str] = []
        if hasattr(resp, "data") and resp.data:
            for m in resp.data:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                if isinstance(mid, str):
                    models.append(mid)
        else:
            for m in resp:
                mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
                if isinstance(mid, str):
                    models.append(mid)
        out: List[str] = []
        seen = set()
        for m in models:
            if m not in seen:
                out.append(m)
                seen.add(m)
            if len(out) >= max_items:
                break
        if not out:
            return [DEFAULT_MODEL]
        return out
    except Exception:
        LOG.exception("Could not list models; using fallback")
        return [DEFAULT_MODEL]

MODEL_OPTIONS = list_models_for_select()

# ---------------- Conversation HTML (srcdoc) ----------------
def generate_conversation_html(messages: Sequence[Dict[str, Any]]) -> str:
    """
    Build full HTML for srcdoc. Contains a unique marker comment used by parent CSS to target this iframe.
    """
    marker = "<!-- RAG_CHAT_IFRAME_MARKER -->"

    css = f"""
    <style>
      :root {{ --center-w: {CENTER_WIDTH_PX}px; --composer-h: {COMPOSER_HEIGHT}px; }}
      html, body {{ margin:0; padding:0; height:100%; background:transparent; font-family: Inter, Helvetica, Arial, sans-serif; color: #e6e6e6; }}
      .rag-outer {{ display:flex; justify-content:center; align-items:flex-start; width:100%; height:100%; box-sizing:border-box; padding:12px 0; background:transparent; }}
      .rag-center {{ width:100%; max-width:var(--center-w); display:flex; flex-direction:column; height:100%; box-sizing:border-box; }}
      .rag-messages {{ flex:1 1 auto; overflow-y:auto; padding:18px 14px 18px 14px; box-sizing:border-box; scroll-behavior:smooth; }}
      .rag-bubble {{ margin:8px 0; padding:12px 14px; border-radius:12px; max-width:86%; line-height:1.45; font-size:15px; word-break:break-word; }}
      .rag-bubble.user {{ margin-left:auto; background:#1f2326; color:#fff; }}
      .rag-bubble.assistant {{ margin-right:auto; background:#f6f7f8; color:#111; }}
      .rag-meta {{ font-size:12px; color:#7f8b91; margin-top:8px; }}
      .rag-attachments {{ display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }}
      .rag-attachment-pill {{ background:#2a2a2a; color:#ddd; padding:6px 10px; border-radius:12px; font-size:13px; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
      pre, code {{ white-space:pre-wrap; word-break:break-word; }}

      /* Dark scrollbars inside iframe */
      .rag-messages::-webkit-scrollbar {{ width:12px; }}
      .rag-messages::-webkit-scrollbar-track {{ background: #0b0b0b; }}
      .rag-messages::-webkit-scrollbar-thumb {{ background: #303033; border-radius:8px; border: 3px solid transparent; background-clip: padding-box; }}
      .rag-messages {{ scrollbar-color: #303033 #0b0b0b; scrollbar-width: thin; }}

      /* Floating "scroll to bottom" button */
      #rag-scroll-to-bottom {{ position:fixed; right:22px; bottom: calc(var(--composer-h) + 22px); width:44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:#1f2326; box-shadow:0 6px 18px rgba(0,0,0,0.45); cursor:pointer; z-index:9999; opacity:0; transform:translateY(8px); transition:opacity .16s, transform .16s; }}
      #rag-scroll-to-bottom.visible {{ opacity:1; transform:translateY(0); }}
      #rag-scroll-to-bottom span {{ font-size:18px; color:#fff; }}
    </style>
    """

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        css,
        "</head><body>",
        marker,
        "<div class='rag-outer'><div class='rag-center'>",
        "<div id='rag-messages' class='rag-messages'>",
    ]

    for m in messages:
        role = m.get("role", "user")
        content = safe_str(m.get("content", ""))
        content_html = "<br>".join(html_lib.escape(line) for line in content.splitlines())
        ts = html_lib.escape(safe_str(m.get("ts", "")))
        cls = "assistant" if role == "assistant" else "user"
        block = f"<div class='rag-bubble {cls}'><div>{content_html or '&nbsp;'}</div>"
        if ts:
            block += f"<div class='rag-meta'>{ts}</div>"
        atts = m.get("_attachments_preview")
        if atts:
            pill_html = "<div class='rag-attachments'>"
            for a in atts:
                pill_html += f"<div class='rag-attachment-pill'>{html_lib.escape(a.get('name',''))}</div>"
            pill_html += "</div>"
            block += pill_html
        block += "</div>"
        parts.append(block)

    parts.append("</div>")  # rag-messages
    parts.append("<div id='rag-scroll-to-bottom' title='Scroll to bottom'><span>&#8593;</span></div>")
    parts.append("</div></div>")

    parts.append(
        """
        <script>
        (function(){
          function q(sel, el) { return (el||document).querySelector(sel); }
          var messages = q('#rag-messages');
          var downBtn = q('#rag-scroll-to-bottom');

          function scrollToBottom() {
            if (!messages) return;
            messages.scrollTop = messages.scrollHeight;
          }

          function handleScroll() {
            if (!messages || !downBtn) return;
            var show = (messages.scrollTop + messages.clientHeight) < (messages.scrollHeight - 60);
            if (show) downBtn.classList.add('visible'); else downBtn.classList.remove('visible');
          }

          if (downBtn) {
            downBtn.addEventListener('click', function(){ scrollToBottom(); downBtn.classList.remove('visible'); });
          }

          if (messages) {
            setTimeout(scrollToBottom, 40);
            messages.addEventListener('scroll', handleScroll);

            var mo = new MutationObserver(function(muts){
              try {
                var nearBottom = (messages.scrollTop + messages.clientHeight) >= (messages.scrollHeight - 120);
                if (nearBottom) {
                  scrollToBottom();
                } else {
                  handleScroll();
                }
              } catch(e) { console.warn(e); }
            });
            mo.observe(messages, { childList: true, subtree: true, characterData: true });
          }
        })();
        </script>
        """
    )

    parts.append("</body></html>")
    return "\n".join(parts)

# ---------------- Render conversation ----------------
layout_container = st.container()
left_col, center_col, right_col = layout_container.columns([1, 2, 1])
conv_placeholder = center_col.empty()

def render_conversation(iframe_pixel_guess: int = IFRAME_FALLBACK_HEIGHT):
    conv = st.session_state.conversation
    for m in conv["messages"]:
        if "_attachments_preview" not in m and m.get("attachments"):
            m["_attachments_preview"] = build_attachment_preview_list(m.get("attachments"))
    html = generate_conversation_html(conv["messages"])

    # Provide large fallback height; parent CSS will clamp to calc(100vh - composer)
    conv_placeholder.empty()
    with conv_placeholder:
        components.html(html, height=iframe_pixel_guess, scrolling=True)

# initial render
render_conversation()

# ---------------- Send callback (synchronous, auto-clear uploader) ----------------
def send_callback():
    conv = st.session_state.conversation
    text = safe_str(st.session_state.get("composer_text", "")).strip()
    files = st.session_state.get(st.session_state.uploader_key)
    api_key = safe_str(st.session_state.get("composer_api_key", "")).strip()

    if not text and not files:
        return

    user_msg = {"role": "user", "content": text, "ts": now_iso()}
    if files:
        user_msg["_attachments_preview"] = build_attachment_preview_list(files)
        user_msg["attachments"] = files
    conv["messages"].append(user_msg)

    assistant_ph = {"role": "assistant", "content": "(thinking…)", "ts": now_iso()}
    conv["messages"].append(assistant_ph)
    render_conversation()

    model_messages = []
    if files:
        previews = build_attachment_preview_list(files)
        attach_text = "\n".join(p["name"] for p in previews)
        if attach_text:
            model_messages.append({"role": "system", "content": "Attached files:\n" + attach_text})
    for m in conv["messages"]:
        if m.get("role") == "assistant" and m.get("content") == "(thinking…)":
            continue
        model_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    model_to_use = st.session_state.get("model_choice", DEFAULT_MODEL)

    use_client = client
    if api_key:
        try:
            use_client = OpenAI(api_key=api_key)
        except Exception:
            LOG.exception("Failed to create OpenAI client with provided API key; falling back to default client")
            use_client = client

    try:
        resp = use_client.chat.completions.create(model=model_to_use, messages=model_messages, temperature=0.0)
        final_text = ""
        if isinstance(resp, dict) and resp.get("choices"):
            c0 = resp["choices"][0]
            if isinstance(c0, dict):
                final_text = c0.get("message", {}).get("content", "") or c0.get("text", "") or ""
        else:
            choices = getattr(resp, "choices", None)
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                if msg:
                    final_text = getattr(msg, "content", "") or ""
        assistant_ph["content"] = final_text or "(no content)"
        assistant_ph["ts"] = now_iso()
        render_conversation()
    except Exception as e:
        LOG.exception("Synchronous completion failed")
        assistant_ph["content"] = f"Error: {str(e)}"
        assistant_ph["ts"] = now_iso()
        render_conversation()
    finally:
        st.session_state["composer_text"] = ""
        # rotate uploader_key to reset uploader widget
        st.session_state.uploader_key = str(uuid.uuid4())

# ---------------- Parent CSS: pin composer & set explicit iframe height (fixes 200px issue) ----------------
st.markdown(
    f"""
    <style>
      html, body, #root, .stApp {{ height:100%; margin:0; padding:0; }}
      .main .block-container {{
         max-width: {CENTER_WIDTH_PX}px;
         margin-left:auto;
         margin-right:auto;
         padding-left:0.5rem;
         padding-right:0.5rem;
         padding-bottom: calc({COMPOSER_HEIGHT}px + 28px);
         box-sizing:border-box;
         height:100vh;
         overflow:hidden; /* page doesn't scroll; iframe handles conversation scrolling */
      }}

      /* Composer pinned bottom center */
      #composer-wrapper {{
         position: fixed;
         left: 50%;
         transform: translateX(-50%);
         bottom: 12px;
         width: 94%;
         max-width: {CENTER_WIDTH_PX}px;
         z-index:2200;
         box-sizing:border-box;
         pointer-events:auto;
      }}
      #composer-inner {{
         display:flex;
         gap:10px;
         align-items:center;
         padding:10px;
         border-radius:12px;
         background: rgba(18,18,18,0.95);
         color: #e6e6e6;
         box-shadow: 0 6px 30px rgba(0,0,0,0.45);
      }}
      #composer-inner textarea {{
         min-height: {COMPOSER_HEIGHT-28}px !important;
         max-height: {COMPOSER_HEIGHT*2}px !important;
         resize: vertical;
      }}
      .stFileUploader {{ margin-top:6px; }}

      /* dark scrollbars */
      ::-webkit-scrollbar {{ width: 12px; height: 12px; }}
      ::-webkit-scrollbar-track {{ background: #070707; }}
      ::-webkit-scrollbar-thumb {{ background: #2f3136; border-radius:8px; border:3px solid transparent; background-clip: padding-box; }}
      html {{ scrollbar-color: #2f3136 #070707; scrollbar-width: thin; }}

      /* ---- CRITICAL: set explicit iframe height so it doesn't collapse to ~200px.
         We target the iframe by the srcdoc marker and force its height to the available viewport
         area above the composer. ---- */
      iframe[srcdoc*="RAG_CHAT_IFRAME_MARKER"] {{
    height: calc(100vh - 240px) !important;   /* 240px = safer reserved area (COMPOSER + chrome) */
    max-height: calc(100vh - 240px) !important;
    min-height: 220px !important;
    display:block;
         margin: 0 auto;
         width: 100% !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <script>
    (function(){
      var composerId = 'composer-wrapper';
      var SELECTOR = 'iframe[srcdoc*="RAG_CHAT_IFRAME_MARKER"]';
      var MIN_HEIGHT = 220;
      var MARGIN = 12;

      function findComposer() { return document.getElementById(composerId); }
      function findIframe() { return document.querySelector(SELECTOR); }

      function setIframeHeight(iframe, desired) {
        var composer = findComposer();
        var composerTop = composer ? composer.getBoundingClientRect().top : window.innerHeight;
        var available = Math.floor(composerTop - MARGIN);
        var h = Math.max(MIN_HEIGHT, Math.min(desired, available));
        iframe.style.setProperty('height', h + 'px', 'important');
        iframe.style.setProperty('max-height', h + 'px', 'important');
      }

      // Listen for postMessage from iframe
      window.addEventListener('message', function(event){
        if (!event.data || event.data.type !== 'RAG_IFRAME_HEIGHT') return;
        var ifr = findIframe();
        if (ifr && ifr.contentWindow === event.source) {
          setIframeHeight(ifr, event.data.height || MIN_HEIGHT);
        }
      });

      // Also run on resize to clamp to composer
      window.addEventListener('resize', function(){
        var ifr = findIframe();
        if (ifr) {
          var cur = parseInt(ifr.style.height || "0") || MIN_HEIGHT;
          setIframeHeight(ifr, cur);
        }
      });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ---------------- Composer UI (pinned) ----------------
center_col.markdown("<div id='composer-wrapper'><div id='composer-inner'>", unsafe_allow_html=True)
col_text, col_right = center_col.columns([6, 2])

col_text.text_area(
    "Message",
    value=st.session_state.get("composer_text", ""),
    key="composer_text",
    label_visibility="collapsed",
    placeholder="Ask anything...",
    height=COMPOSER_HEIGHT,
)

col_text.text_input(
    "OpenAI API key (optional)",
    value=st.session_state.get("composer_api_key", ""),
    key="composer_api_key",
    placeholder="sk-... (optional)",
    label_visibility="collapsed",
    type="password",
)

with col_right:
    if "model_choice" in st.session_state:
        st.selectbox("Model", options=MODEL_OPTIONS, key="model_choice", label_visibility="collapsed")
    else:
        default_idx = MODEL_OPTIONS.index(DEFAULT_MODEL) if DEFAULT_MODEL in MODEL_OPTIONS else 0
        st.selectbox("Model", options=MODEL_OPTIONS, index=default_idx, key="model_choice", label_visibility="collapsed")

    composer_has_text = bool(safe_str(st.session_state.get("composer_text", "")).strip())
    composer_has_files = bool(st.session_state.get(st.session_state.uploader_key))
    disabled_send = not (composer_has_text or composer_has_files)

    st.button("Send", key="composer_send", on_click=send_callback, disabled=disabled_send)

    st.file_uploader(
        "Attach files",
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
        label_visibility="collapsed",
    )

center_col.markdown("</div></div>", unsafe_allow_html=True)

# ---------------- Ctrl/Cmd+Enter shortcut ----------------
st.markdown(
    """
    <script>
    (function () {
      document.addEventListener('keydown', function (e) {
        var isEnter = (e.key === 'Enter' || e.keyCode === 13);
        var mod = e.ctrlKey || e.metaKey;
        if (!(isEnter && mod)) return;
        var ta = document.activeElement;
        if (!ta || ta.tagName !== 'TEXTAREA') return;
        var buttons = Array.from(document.querySelectorAll('button'));
        for (var i = buttons.length - 1; i >= 0; --i) {
          var b = buttons[i];
          try {
            var txt = (b.innerText || b.textContent || '').trim().toLowerCase();
            if (txt === 'send' || txt.indexOf('send') !== -1) {
              if (!b.disabled) {
                b.click();
              }
              e.preventDefault();
              return;
            }
          } catch (ex) {}
        }
      }, false);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# Done.
