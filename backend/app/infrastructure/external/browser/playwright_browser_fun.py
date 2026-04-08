#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Playwright 浏览器结构化提取脚本。"""

GET_PAGE_STRUCTURED_DATA_FUNC = """() => {
    const isVisible = (element) => {
        if (!element) return false;
        const rect = element.getBoundingClientRect();
        const style = window.getComputedStyle(element);
        return !(
            rect.width === 0 ||
            rect.height === 0 ||
            rect.bottom < 0 ||
            rect.top > window.innerHeight ||
            rect.right < 0 ||
            rect.left > window.innerWidth ||
            style.display === 'none' ||
            style.visibility === 'hidden' ||
            style.opacity === '0'
        );
    };

    const normalizeText = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
    const truncate = (value, maxChars = 240) => {
        const normalized = normalizeText(value);
        return normalized.length > maxChars ? normalized.slice(0, maxChars - 3) + '...' : normalized;
    };

    const pickMainRoot = () => {
        const candidates = Array.from(document.querySelectorAll('main, article, [role="main"], .markdown-body, .article, .post, .content'));
        let best = null;
        let bestScore = -1;

        for (const candidate of candidates) {
            if (!isVisible(candidate)) continue;
            const text = normalizeText(candidate.innerText);
            const paragraphCount = candidate.querySelectorAll('p').length;
            const headingCount = candidate.querySelectorAll('h1, h2, h3').length;
            const score = text.length + paragraphCount * 120 + headingCount * 80;
            if (score > bestScore) {
                bestScore = score;
                best = candidate;
            }
        }

        if (best) return best;
        return document.body;
    };

    const mainRoot = pickMainRoot();
    const mainHeadingElement = Array.from(document.querySelectorAll('h1, h2')).find((element) => isVisible(element));
    const mainHeading = normalizeText(mainHeadingElement ? mainHeadingElement.innerText : '');
    const mainText = truncate(mainRoot ? mainRoot.innerText : '', 4000);
    const mainHtml = mainRoot ? mainRoot.outerHTML : '';

    const actionableElements = [];
    const actionableNodes = Array.from(
        document.querySelectorAll('button, a, input, textarea, select, [role="button"], [tabindex]:not([tabindex="-1"])')
    );
    let actionableIndex = 0;
    for (const element of actionableNodes) {
        if (!isVisible(element)) continue;
        const tag = String(element.tagName || '').toLowerCase();
        const text = truncate(
            element.innerText ||
            element.value ||
            element.getAttribute('aria-label') ||
            element.getAttribute('placeholder') ||
            element.getAttribute('title') ||
            element.getAttribute('alt') ||
            '',
            120
        );
        const selector = `[data-manus-id="manus-element-${actionableIndex}"]`;
        element.setAttribute('data-manus-id', `manus-element-${actionableIndex}`);
        actionableElements.push({
            index: actionableIndex,
            tag,
            text,
            role: normalizeText(element.getAttribute('role') || ''),
            selector,
        });
        actionableIndex += 1;
    }

    const cardCandidates = [];
    const seenUrls = new Set();
    const linkNodes = Array.from(document.querySelectorAll('a[href]'));
    for (const link of linkNodes) {
        if (!isVisible(link)) continue;
        const href = normalizeText(link.href);
        const title = truncate(link.innerText || link.getAttribute('title') || link.getAttribute('aria-label') || '', 160);
        if (!href || !title || seenUrls.has(href)) continue;
        seenUrls.add(href);

        const container = link.closest('article, li, section, div');
        let summary = '';
        if (container && isVisible(container)) {
            const containerText = normalizeText(container.innerText);
            summary = truncate(containerText.replace(title, '').trim(), 240);
        }
        cardCandidates.push({
            index: cardCandidates.length,
            title,
            summary,
            url: href,
            tags: [],
        });
        if (cardCandidates.length >= 12) break;
    }

    const scrollHeight = Math.max(
        document.documentElement.scrollHeight || 0,
        document.body ? document.body.scrollHeight || 0 : 0,
    );
    const scrollTop = window.scrollY || document.documentElement.scrollTop || 0;
    const viewportHeight = window.innerHeight || 0;
    const maxScrollable = Math.max(scrollHeight - viewportHeight, 0);
    const scrollProgress = maxScrollable > 0 ? Math.min((scrollTop / maxScrollable) * 100, 100) : 100;

    return {
        url: window.location.href || '',
        title: document.title || '',
        main_heading: mainHeading,
        main_text: mainText,
        main_html: mainHtml,
        body_text: truncate(document.body ? document.body.innerText : '', 5000),
        card_candidates: cardCandidates,
        actionable_elements: actionableElements,
        paragraph_count: document.querySelectorAll('p').length,
        heading_count: document.querySelectorAll('h1, h2, h3').length,
        form_count: document.querySelectorAll('form').length,
        link_count: linkNodes.length,
        scroll_progress: scrollProgress,
        should_continue_scrolling: scrollProgress < 92 && scrollHeight > viewportHeight * 1.2,
    };
}
"""

INJECT_CONSOLE_LOGS_FUNC = """() => {
    window.console.logs = window.console.logs || [];
    const originalLog = console.log;
    if (!console.__lingxiWrappedLog) {
        console.log = (...args) => {
            window.console.logs.push(args.join(" "));
            originalLog.apply(console, args);
        };
        console.__lingxiWrappedLog = true;
    }
}"""
