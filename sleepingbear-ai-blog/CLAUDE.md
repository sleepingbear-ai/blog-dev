# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Hugo static site for the SleepingBear blog — a bilingual (English + Chinese) blog about AI, reading, and thoughts. Content is authored in Markdown; the site is rendered with the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme. Requires Hugo **extended** (developed with v0.162.1).

## Writing Blog Posts
### Language

Use Chinese whenever possible, but use English words and phrases where natural. For example, common English terms well-known to Chinese people can be written in English directly, such as: company names "OpenAI", "Anthropic"; AI technologies "RAG", "LLM", "Agents", "Prompts", "Deep Research", "Multi-Agent", "Single-Agent".

### Writing Style

Make each post simple, concise, short, direct, and easy to understand. Each post ideally should be less than 5 pages.

### Format

Prefer Markdown format.

### Web Links

Use Web Links whenever it makes sense.

### Images

Use Images whenever it makes sense, but not too many to be distracting. Prefer using real images instead of ascii diagrams.

## Commands

```bash
hugo server -D          # Local dev server with live reload; -D includes draft posts
hugo new content posts/my-post.md   # Scaffold a new post from archetypes/default.md
hugo                    # Build the site into public/
```

New posts are created as `draft = true` (see `archetypes/default.md`). Set `draft = false` before publishing, or they won't appear in a normal `hugo` build.

## Deployment architecture

This repo (`blog-dev`) is the **source**. The built site lives in `public/`, which is a **separate, nested git repository** pointing to `github.com/sleepingbear-ai/sleepingbear-ai.github.io` (the GitHub Pages site at `baseURL` in `hugo.toml`). There is no CI workflow — publishing is manual and two-step:

1. From the repo root, run `hugo` to regenerate `public/`.
2. `cd public`, then commit and push there to publish to GitHub Pages.

Because `public/` is its own repo, `git status` at the root shows `public` as a single changed entry (a gitlink-like pointer) rather than individual built files. Commit source changes (content, config, theme) at the root; commit generated output separately inside `public/`. Never hand-edit files in `public/` — they are overwritten on every build.

## Layout notes

- `content/posts/` — blog posts. Front matter is TOML (`+++` delimiters), per PaperMod/Hugo defaults.
- `themes/hugo-PaperMod/` — vendored clone of the theme (its own `.git`, tracking `adityatelange/hugo-PaperMod`). Prefer overriding in the project's top-level `layouts/`, `assets/`, `static/`, and `i18n/` rather than editing theme files, so theme updates don't clobber changes.
- `hugo.toml` — minimal config (baseURL, locale, title, theme).
