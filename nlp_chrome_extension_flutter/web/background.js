/*var htmlCode;

chrome.tabs.onUpdated.addListener(function (tabId, info) {
    if(info.status === 'complete') {
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            func: DOMtoString,
        }, function(result) {
            if (chrome.runtime.lastError) {
                console.error(chrome.runtime.lastError.message);
            } else {
                htmlCode = result[0].result;
                console.log(htmlCode);
            }
        });
    }
});

function DOMtoString(selector) {
    if (selector) {
        selector = document.querySelector(selector);
        if (!selector) return "ERROR: querySelector failed to find node"
    } else {
        selector = document.documentElement;
    }
    return selector.outerHTML;
}

function getHtmlCode() {
    return htmlCode;
}*/