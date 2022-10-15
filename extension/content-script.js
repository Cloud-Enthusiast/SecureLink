let a = document.getElementsByTagName('a');
let results = [];

for (var idx= 0; idx < a.length; ++idx){
    if (!(a[idx].href.includes("google.")) && a[idx].href !== "")
        results.push(a[idx].href)
}

console.log(results)