function getHtmlCode() {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        var activeTab = tabs[0];
        var activeTabId = activeTab.id;

       return chrome.scripting.executeScript({
           target: { tabId: activeTabId },
           func: DOMtoString,
       }, function(result) {
           if (chrome.runtime.lastError) {
               console.error(chrome.runtime.lastError.message);
           } else {
               chrome.storage.local.set({'data': result[0].result}, function() {
               });
           }
       });
    });
}

function DOMtoString(selector) {
    if (selector) {
        selector = document.querySelector(selector);
        if (!selector) return "ERROR: querySelector failed to find node"
    } else {
        selector = document.documentElement;
    }
    return selector.outerHTML;
}