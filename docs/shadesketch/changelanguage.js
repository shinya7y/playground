window.onload = function() {
  var wDef = (navigator.browserLanguage || navigator.language || navigator.userLanguage).substr(0,2);
  langSet(wDef);
}
function langSet(argLang){
  if (argLang != "ja") {
    argLang = "en";
  }
  document.getElementById(argLang + "LangRadio").checked = true;

  var elm = document.getElementsByClassName("langCng");
  for (let i = 0; i < elm.length; i++) {
    if(elm[i].getAttribute("lang") == argLang){
      elm[i].style.display = '';
    }
    else{
      elm[i].style.display = 'none';
    }
  }
}