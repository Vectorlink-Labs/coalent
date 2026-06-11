# Coalent — Brand Guide

> Open [`showcase.html`](./showcase.html) in a browser for the full brand sheet.

## The mark — “Focus-C”

Coalent makes scattered, ever-changing context **cogent** — coherent, current,
decision-ready. The mark is a **"C"** drawn as an open ring gathering inward to a
bright focal core (the materialized, fresh understanding), with three cyan
provenance nodes on the arc. It reads as a letterform, a lens (focus = "cogent"),
and a graph at once — and stays legible down to a 20px favicon.

## Logo files

| File | Use |
|---|---|
| `wordmark.svg` | primary lockup (mark + “coalent”), light backgrounds |
| `wordmark-dark.svg` | wordmark for dark surfaces |
| `logomark.svg` | the mark alone (app icon, social avatar) |
| `favicon.svg` | dark-tile favicon, optimized for tiny sizes |
| `tokens.css` / `tokens.json` | brand color + type tokens for the site/app |

## Color system

| Token | Hex | Role |
|---|---|---|
| Ink | `#0B1020` | text, dark surfaces |
| Coalent Indigo | `#6366F1` | primary brand |
| Indigo Deep | `#4F46E5` | hover/active, depth |
| Signal Cyan | `#22D3EE` | freshness / live accent |
| Mint (alt) | `#2DD4BF` | secondary accent |
| Paper | `#FBFCFE` | light surfaces |
| Slate | `#475569` | secondary text |

**Signature gradient:** `linear-gradient(90deg, #6366F1, #22D3EE)` — *depth →
clarity, source → understanding, stale → fresh.* It drives the mark, primary
buttons, links, and section accents.

On dark surfaces, lift the indigo stop to `#818CF8` for contrast (see the dark
wordmark in the showcase).

## Typography

- **Wordmark & UI:** Inter (fallback: Segoe UI / system-ui), weight 600 for the
  wordmark, −0.6 tracking. Lowercase “coalent”.
- **Code/mono:** ui-monospace / Consolas.

## Usage

- **Clear space:** keep padding ≥ the height of the core dot around the mark.
- **Min size:** icon legible to 20px; below that, drop the arc nodes (use the
  C + core only — see the favicon row in the showcase).
- **Don’t:** recolor outside the palette, add shadows/bevels, stretch, or place
  the gradient mark on a busy photo without a solid tile behind it.
