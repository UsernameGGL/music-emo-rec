//妫€鏌ユ槸鍚﹀畨瑁呰蒋浠�
var xiamiInstalled=false;
_xiamitoken = typeof(_xiamitoken)=='undefined' ? '' : _xiamitoken;

var getBrowser = function(){
	var s = navigator.userAgent.toLowerCase();
	var a = new Array("msie", "firefox", "safari", "opera", "netscape");
	for(var i = 0; i < a.length; i ++){
		if(s.indexOf(a[i]) != -1){
			return a[i];
		}
	}
	return "other";
};

try{
	if(!xiamiInstalled){
		var obj=new ActiveXObject("XiaMiplugin.xiamistart");
		if(obj){
			xiamiInstalled=true;
			delete obj
		}
	}
}catch(e){}

function runxiami(){
	if(xiamiInstalled || getBrowser() != 'msie'){
		return true;
	}else{
		window.location='/software/shark';
		return false;
	}
};

function runxiami_p(p){
	if(xiamiInstalled || getBrowser() == 'firefox'){
		window.location= p;
	}else{
		window.location='/software';
	}
};

//閫夋嫨name 鐨刢heck box
function selectAll(name){
	var arr = $('input[name='+name+']');
	for(var i=0;i<arr.length;i++){
		if(arr[i].disabled == false){
			arr[i].checked = true;
		}
	}
};

//鍙嶉€�
function inverse(name){
	var arr = $('input[name='+name+']');
	for(var i=0;i<arr.length;i++){
		if(arr[i].disabled == false){
			arr[i].checked = ! arr[i].checked;
		}
	}
};

//鑾峰彇閫夋嫨鐨勫€�
// function getSelectValues(name){
// 	var sValue = '';
// 	var tmpels = $('input[name='+name+']');
// 	for(var i=0,j=0;i<tmpels.length;i++){
// 		if (j==100) break;
// 		if(tmpels[i].checked || (tmpels[i].type=='hidden' && tmpels[i].defaultChecked)){
// 			if(sValue == ''){
// 				sValue = tmpels[i].value;
// 			}else{
// 				sValue = sValue + "," + tmpels[i].value;
// 			}
// 			j++;
// 		}
// 	}
// 	return sValue;
// };

//鑾峰彇鍊�
function getValues(name){
	var sValue = "";
	var tmpels = $('input[name='+name+']');
	for(var i=0;i<tmpels.length;i++){
		if(sValue == ""){
			sValue = tmpels[i].value;
		}else{
			sValue = sValue + "," + tmpels[i].value;
		}
	}
	return sValue;
};
var getFlashVersion = function(){
    var nav = navigator, version = 0, flash;
    if(nav.plugins && nav.mimeTypes.length){
        flash = navigator.plugins["Shockwave Flash"];
        if(flash) {
            version = flash.description.replace(/.*\s(\d+\.\d+).*/, "$1");
        }
    }else{
        try{
            flash = new window.ActiveXObject("ShockwaveFlash.ShockwaveFlash");
            if(flash){
                version = flash.GetVariable("$version")
            }
        } catch(e){}
    }
    if(version !== 0 ){
    	var fv = version.match(/\d+/g);
    	if(fv.length > 0){
    		var v = fv.join('.')
    		getFlashVersion = function(){
    			return v
    		};
    		return v;
    	}
    }
    return version;
};
function getInternetExplorerVersion(){
	var rv = -1, ua = navigator.userAgent;
	if (navigator.appName == 'Microsoft Internet Explorer'){
		var re  = new RegExp("MSIE ([0-9]{1,}[\.0-9]{0,})");
		if (re.exec(ua) != null)
			rv = parseFloat( RegExp.$1 );
	}
	else if (navigator.appName == 'Netscape'){
		var re  = new RegExp("Trident/.*rv:([0-9]{1,}[\.0-9]{0,})");
		if (re.exec(ua) != null)
			rv = parseFloat( RegExp.$1 );
	}
	return rv;
}
function getOperaVersion(){
	var rv = -1, ua = navigator.userAgent;
	if (navigator.appName == 'Opera'){
		var re  = new RegExp("Opera\/.*Version\/([0-9]{1,}[\.0-9]{0,})");
		if (re.exec(ua) != null)
			rv = parseFloat( RegExp.$1 );
	}
	return rv;
}
// 鑾峰彇 flash core
function thisMovie(name) {
	var ie = getInternetExplorerVersion();
	if( ie != -1 && ie <= 9){
		thisMovie = function(name){
			return window[name];
		}
		return window[name];
	}else{
		thisMovie = function(name){
			return document.getElementById(name);
		}
		return document.getElementById(name)
	}
};


var playerDialog;
function openMusicPlayer(str){
	//鏇存敼鎾斁鍣ㄩ珮搴� 绔欏banner
	var reg = str.indexOf('out=1');
	if(reg != -1  && screen.height >= 640){
		playerDialog = window.open("//emumo.xiami.com/song/play?ids="+str,"xiamiMusicPlayer",'width=754,height=637');
		return;
	}
	transclick(str, 'open');
	var o = getOperaVersion(), i = getInternetExplorerVersion();
	if((i == -1 || i > 6) && (o == -1 || o > 15 )){
		playerDialog = window.open("//emumo.xiami.com/play?ids="+str+"#open","xiamiMusicPlayer");
	}else{
	//鍏朵粬鍦版柟浣跨敤
		playerDialog = window.open("//emumo.xiami.com/song/play?ids="+str+"#open","xiamiMusicPlayer",'width=754,height=557');
	}
}
function openOldMusicPlayer(str) {
    transclick(str, 'openold');
    playerDialog = window.open("//emumo.xiami.com/song/play?ids=" + str + "#open", "xiamiMusicPlayer", 'width=754,height=557');
}

function openPlayer(str){
	var o = getOperaVersion(), i = getInternetExplorerVersion();
	if((i == -1 || i > 6) && (o == -1 || o > 15 )){
		playerDialog = window.open("//emumo.xiami.com/play?ids="+str,"xiamiMusicPlayer");
	}else{
		window.open("//emumo.xiami.com/song/play?ids="+str, "xiamiMusicPlayer", 'scrollbars,width=720,height=645');
	}
};

//function player_focus(){
//	if(!playerDialog) playerDialog = window.open('',"xiamiMusicPlayer",'width=754,height=656');
//	playerDialog.focus();
//}
function transclick(val, type){
	var n = 'log_' + (new Date()).getTime();
	var url = '//emumo.xiami.com/blank.gif';
	var data = [
		'f=seiya',
		't=' + type,
		'fv=' + getFlashVersion(),
        'bm_an=' + navigator.appName,
        'v=' + val,
		'_=' + (new Date()).getTime()
	];
	var req = window[n] = new Image();
	req.onload = req.onerror= function(){
		window[n] = null;
	};
	req.src = url + '?' + data.join('&')
	req = null;
};

function addSongs(str) {
    var playerTrigger = thisMovie('trans');
    if (typeof(playerTrigger.addSongs) === "function") {
        playerTrigger.addSongs(str);
        transclick(str, "add");
    } else {
        setTimeout(function() {
            if (typeof(playerTrigger.addSongs) === "function") {
                playerTrigger.addSongs(str);
                transclick(str, "add");
            } else {
                openOldMusicPlayer(str);
            }
        }, 500);
    }
}

function playsongs(checkname,type_name,type_id,cat_id){
	var ids = getSelectValues(checkname);
	if(ids == ''){alert("娌℃湁姝屾洸鍙互鎾斁!");return;}
	if(!type_name)type_name  ='default';
	if(!type_id)type_id = 0;
	if(!cat_id)cat_id = 0;
	if(cat_id){
		addSongs(escape("/song/playlist/id/"+ids+"/object_name/"+type_name+"/object_id/"+type_id+"/cat_id/"+cat_id));
	} else {
		addSongs(escape("/song/playlist/id/"+ids+"/object_name/"+type_name+"/object_id/"+type_id));
	}
};

function playall(checkname,type_name,type_id) {
	var ids = getValues(checkname);
	if(ids == ''){alert("娌℃湁姝屾洸鍙互鎾斁!");return;}
	if(!type_name)type_name  ='default';
	if(!type_id)type_id = 0;
	addSongs(escape("/song/playlist/id/"+ids+"/object_name/"+type_name+"/object_id/"+type_id));
};

function playsongsignore(checkname,type_name,type_id){
	var ids = getSelectValues(checkname);
	if(ids == ''){alert("娌℃湁姝屾洸鍙互鎾斁!");return;}
	if(!type_name) type_name  ='default';
	if(!type_id) type_id = 0;
	addSongs(escape("/song/playlist/id/"+ids+"/object_name/"+type_name+"/object_id/"+type_id));
};

//type_name : collect , album
function play(song_id,type_name,type_id){
	if(!type_name) type_name  ='default';
	if(!type_id)type_id = 0;
	addSongs(escape("/song/playlist/id/"+song_id+"/object_name/"+type_name+"/object_id/"+type_id));
};

function playalbum(album_id){
	addSongs(escape("/song/playlist/id/"+album_id+"/type/1"));
};

function playartist(artist_id){
	addSongs(escape("/song/playlist/id/"+artist_id+"/type/2"));
};

function playcollect(list_id){
	addSongs(escape("/song/playlist/id/"+list_id+"/type/3"));
};

function playFriendRecommend(user_id){
	addSongs(escape("/song/playlist/id/"+user_id+"/type/4"));
};

function playdefault() {
	addSongs(escape("/song/playlist/id/1/type/9"));
}
/*绔欏鎾斁*/
function playsongsout(checkname,type_name,type_id,cat_id){
	var ids = getSelectValues(checkname);
	if(ids == ''){alert("娌℃湁姝屾洸鍙互鎾斁!");return;}
	if(!type_name)type_name  ='default';
	if(!type_id)type_id = 0;
	if(!cat_id)cat_id = 0;
	if(cat_id){
		addSongs(escape("/song/playlist/id/"+ids+"/object_name/"+type_name+"/object_id/"+type_id+"/cat_id/"+cat_id)+'&out=1');
	} else {
		addSongs(escape("/song/playlist/id/"+ids+"/object_name/"+type_name+"/object_id/"+type_id)+'&out=1');
	}
};

function playallout(checkname,type_name,type_id) {
	var ids = getValues(checkname);
	if(ids == ''){alert("娌℃湁姝屾洸鍙互鎾斁!");return;}
	if(!type_name)type_name  ='default';
	if(!type_id)type_id = 0;
	addSongs(escape("/song/playlist/id/"+ids+"/object_name/"+type_name+"/object_id/"+type_id)+'&out=1');
};

function playsongsignoreout(checkname,type_name,type_id){
	var ids = getSelectValues(checkname);
	if(ids == ''){alert("娌℃湁姝屾洸鍙互鎾斁!");return;}
	if(!type_name) type_name  ='default';
	if(!type_id) type_id = 0;
	addSongs(escape("/song/playlist/id/"+ids+"/object_name/"+type_name+"/object_id/"+type_id)+'&out=1');
};

//type_name : collect , album
function playout(song_id,type_name,type_id){
	if(!type_name) type_name  ='default';
	if(!type_id)type_id = 0;
	addSongs(escape("/song/playlist/id/"+song_id+"/object_name/"+type_name+"/object_id/"+type_id)+'&out=1');
};

function playalbumout(album_id){
	addSongs(escape("/song/playlist/id/"+album_id+"/type/1")+'&out=1');
};

function playartistout(artist_id){
	addSongs(escape("/song/playlist/id/"+artist_id+"/type/2")+'&out=1');
};

function playcollectout(list_id){
	addSongs(escape("/song/playlist/id/"+list_id+"/type/3")+'&out=1');
};

function playFriendRecommendout(user_id){
	addSongs(escape("/song/playlist/id/"+user_id+"/type/4")+'&out=1');
};

function playdefaultout() {
	addSongs(escape("/song/playlist/id/1/type/9")+'&out=1');
}
/*	*/

var ajaxText = '<div class="dialog_content"><p class="loading">铏惧皬绫虫涓烘偍鍦ㄥ鐞嗘暟鎹�, 璇风◢鍊�...</p></div><a href="javascript:;" title="" onclick="closedialog();" class="Closeit">鍏抽棴</a>';

//ie6涓嬮珮搴︾殑闂
var dialogbg=function(){
	if(getBrowser() == 'msie') {
		$('.dialog_sharp').height($('.dialog_main').height());
	}
};

var myjqmOnShow = function(hash){
	hash.w.show();
	dialogbg();
};

var myjqmOnLoad = function(hash){
	dialogbg();
};

function showDialog(url,target,ajaxText){
	if(!target) target = "div.dialog_main";
	if(!ajaxText) ajaxText = '<div class="dialog_content"><p class="loading">铏惧皬绫虫涓烘偍鍦ㄥ鐞嗘暟鎹�, 璇风◢鍊�...</p></div><a href="javascript:;" title="" onclick="closedialog();" class="Closeit">鍏抽棴</a>';
	if(!url){
        $('#dialog_clt .dialog_main').html(ajaxText);
    }else{
        url += (url.match(/\?/) ? "&" : "?") + "_xiamitoken="+_xiamitoken;
    }
	$('#dialog_clt').jqm({
		ajax:url,
		modal:true,
		toTop:true,
		target: target,
		ajaxText: ajaxText,
		onShow:myjqmOnShow,
		onLoad:myjqmOnLoad
	}).jqDrag('.jqDrag').jqmShow();
};

/** showAlert noyobo 2013骞�3鏈�16鏃� 20:35:04
 @param  msg : string 鎻愮ず鍐呭
 @param  title: string [option] 鎻愮ず鏍囬
 @param  type: string ok|error 鎻愮ず绫诲瀷
*/
function showAlert( msg, title, type ){
	var html = "";
	switch( type ){
		case "ok": html = '<div class="dialog_main">	<h3>{title}</h3><div class="dialog_content"><div id="success_msg" class="alert alert_ok"><strong>{msg}</strong></div></div><a href="javascript:void(0);" onclick="closedialog();" class="Closeit">鍏抽棴</a></div><div class="dialog_acts"><input type="button" class="bt_sub2" value="纭� 瀹�"  onclick="closedialog();"/></div>';
			break;
		case "error": html = '<div class="dialog_main"><h3>{title}</h3><div class="dialog_content"><div id="success_msg" class="alert alert_error"><strong>{msg}</strong></div></div><a href="javascript:void(0);" onclick="closedialog();" class="Closeit">鍏抽棴</a>	</div><div class="dialog_acts"><input type="button" class="bt_sub2" value="纭� 瀹�"  onclick="closedialog();"/></div>';
			break;
		default:
			html = '<div class="dialog_main"><h3>鎻愮ず</h3><div class="dialog_content"><div id="success_msg"><strong>{msg}</strong></div></div><a href="javascript:void(0);" onclick="closedialog();" class="Closeit">鍏抽棴</a></div><div class="dialog_acts"><input type="button" class="bt_sub2" value="纭� 瀹�"  onclick="closedialog();"/></div>';
	};
	html = html.replace(/{msg}/g, msg);
	if (typeof(title) != "undefined" ) html = html.replace(/{title}/g, title);

	$('#dialog_clt .dialog_main').html( html );
	$('#dialog_clt').jqm({
		ajax:null,
		modal:true,
		toTop:true
	}).jqDrag('.jqDrag').jqmShow();
	dialogbg();
};
/**
* showConfirm noyobo 2013骞�3鏈�16鏃� 21:20:30
* @param	msg 鎻愮ず鍐呭
* @param	callback : function
*/
function showConfirm(msg, callback) {
	var html = '<div class="dialog_main"><h3>纭鎻愮ず</h3><div class="dialog_content"><div id="success_msg"><strong>{msg}</strong></div></div><a href="javascript:void(0);" onclick="closedialog();" class="Closeit">鍏抽棴</a></div><div class="dialog_acts"><input type="button" class="bt_sub2" value="纭� 瀹�" />&nbsp;&nbsp;&nbsp;&nbsp;<input type="button" class="bt_cancle2" value="鍙� 娑�" /></div>';
		html = html.replace(/{msg}/g, msg);

	$('#dialog_clt .dialog_main').html( html );
	$('#dialog_clt').jqm({
		ajax:null,
		modal:true,
		toTop:true
	}).jqDrag('.jqDrag').jqmShow().find('div.dialog_main')
	.end().find(':button:visible').click(function(){
		$('#dialog_clt').jqmHide();
		if(this.className == 'bt_sub2'){
			if (typeof callback == 'function') callback();
		}
	});
	dialogbg();
};

function collect(id){
	var url = '/song/collect/id/'+ encodeURIComponent(id);
	showDialog(url);
};

function zoneablum(id){
	var url = '/zone/addablum/id/'+encodeURIComponent(id);
	showDialog(url);
}

function zoneCatergory(id,type){
	var url = '/zone/editsort/id/'+encodeURIComponent(id)+'/type/'+encodeURIComponent(type)+'/?'+Math.random();
	showDialog(url);
}

function tag(id,type){
	var url = '/music/tag/type/'+encodeURIComponent(type)+'/id/'+ encodeURIComponent(id);
	showDialog(url);
};

function tagedit(id,type){
	var url = '/music/tagedit/type/'+encodeURIComponent(type)+'/id/'+ encodeURIComponent(id);
	showDialog(url);
}

function closedialog(){
	$('#dialog_clt').jqmHide();
};

function insertsongs(){
	var url = '/search/popsongs?id=x';
	$('#dialog').jqm({
	ajax:url,
	modal:true,
	toTop:true,
	target: 'div.pop_message',
	ajaxText: ajaxText,
	onShow:myjqmOnShow,
	onLoad:myjqmOnLoad
	}).jqDrag('.jqDrag').jqmShow();
};

function search_songs(search_result,key){
	var url = '/search/searchpopsongs';
	var pars = 'key=' + encodeURIComponent(key);
	var myAjax = new Ajax.Updater(
		search_result, // 鏇存柊鐨勯〉闈㈠厓绱�
		url, // 璇锋眰鐨刄RL
		{method: 'post',parameters: pars,evalScripts: true}
	);
};

//涓嬭浇鍗曟洸
function download(id){
	if (!$.cookie('user')) {
		showDialog('/member/poplogin');
		return;
	}
  prepareZipx('song', encodeURIComponent(id));return;
	var url = '//emumo.xiami.com/download/pay?id='+encodeURIComponent(id);
	//showDialog(url);
    window.open(url);
};

//chrome鏂扮増bug锛宱nclick鏃犳硶浣跨敤download鍛藉悕鐨勫嚱鏁�
function xm_download(id) {
	if (!$.cookie('user')) {
		showDialog('/member/poplogin');
		return;
	}
  prepareZipx('song', encodeURIComponent(id));return;
	var url = '//emumo.xiami.com/download/pay?id='+encodeURIComponent(id);
    //showDialog(url);
    window.open(url);
}

//鎺ㄥ箍鐨勪笅杞藉崟鏇�
function promotion_download(id,type,pid, ele){

	if (type && type == 1) {
        var pare = ele.parentNode.parentNode.parentNode.parentNode;
        if (pare) {
            var data = pare.getAttribute('data-json');
            data = JSON.parse(decodeURIComponent(data));
            data.id = id;
						console.log(data);
						downloadToPC('song', id);
            // selectDownlodQuality(data);
            return;
            // var downloadstatus = pare.getAttribute('data-downloadstatus');
            // if (downloadstatus) {
            //     if (downloadstatus == '0') {
            //         checkSongPermission('download')
            //         return;
            //     } else if (downloadstatus == '2') {
            //         buyMusic('song', id, '涓嬭浇');
            //         return;
            //     }
            // }
        }
    }

	// var url = '//emumo.xiami.com/download/pay?id='+ encodeURIComponent(id) +'&ptype='+type +'&pid='+pid;;
 //    //showDialog(url);
 //    window.open(url);
};

//涓嬭浇涓撹緫
function downloadalbum(id,type, me){
	if (!$.cookie('user')) {
		showDialog('/member/poplogin');
		return;
	}
	if (me) {
      needpay = me.getAttribute('data-needpay'); // 0 涓嶉渶瑕佷粯璐�, 1 闇€瑕佷粯璐�
      playstatus = me.getAttribute('data-playstatus'); // 0 涓嶆彁渚涙湇鍔�, 1 鍏嶈垂, 2 浠樿垂
      downloadstatus = me.getAttribute('data-downloadstatus'); // 0 涓嶆彁渚涙湇鍔�, 1 鍏嶈垂, 2 浠樿垂

      if (downloadstatus == '0') {
          checkAlbumPermission('download');
          return;
      }

      if (downloadstatus && downloadstatus == '2') {
          buyMusic('album', id, '涓嬭浇');
          return;
      }
  }
  downloadToPC('album', id);
  //prepareZip();
  return;
  //selectDownlodQuality();return;
	//var url = '//emumo.xiami.com/download/pay?id=' + encodeURIComponent(id) + '&type=album';
	//showDialog(url);
    window.open(url);
};
//涓嬭浇涓撹緫2
function downloadalbum2(id,type){
	if (!$.cookie('user')) {
		showDialog('/member/poplogin');
		return;
	}
  prepareZipx('album', id);return;
	var url = '//emumo.xiami.com/download/song?id=' + encodeURIComponent(id) + '&type=album';
	$('#dialog_clt').jqm({
	ajax:url,
	modal:true,
	toTop:true,
	target: 'div.dialog_main',
	ajaxText: ajaxText,
	onShow:myjqmOnShow,
	onLoad:myjqmOnLoad
	}).jqDrag('.jqDrag').jqmShow();
}

function downloadLosslessAlbum(id){
    var url = '/download/pay?id=' + encodeURIComponent(id) + '&type=album&lossless=1&tradition=1';
    window.open(url);
}

//鍙備笌涓婚姝屽崟
function joinsub(id,type){
	var url = '/collect/joinsub?id=' + encodeURIComponent(id)+"&type="+encodeURIComponent(type);
	showDialog(url);
};

//涓嬭浇姝屽崟
function downloadcollect(id,type){
	if (!$.cookie('user')) {
		showDialog('/member/poplogin');
		return;
	}
	// prepareZipx('collect', encodeURIComponent(id), '')
	downloadToPC('collect', id);
  return;
	var url = '//emumo.xiami.com/download/pay?id=' + encodeURIComponent(id) + '&type=collect';
    //showDialog(url);
    window.open(url);
};

function promotion_downloadsongs(ids,type,pid) {
	if (!$.cookie('user')) {
		showDialog('/member/poplogin');
		return;
	}
	var url = '//emumo.xiami.com/download/pay';
	var id = getSelectValues_2(ids);
	if(id == ''){
		alert("娌℃湁璧勬簮鍙互涓嬭浇锛�");
		return;
	}
  return;
	var url = url+'?id=' + encodeURIComponent(id)+'&ptype='+type +'&pid='+pid;
    //showDialog(url);
    window.open(url);
};

//涓嬭浇澶氶姝屾洸
function downloadsongs(ids){
	if (!$.cookie('user')) {
		showDialog('/member/poplogin');
		return;
	}
	var url = '//emumo.xiami.com/download/pay';
	var id = getSelectValues_2(ids);
	if(id == ''){alert("娌℃湁璧勬簮鍙互涓嬭浇锛�");return;}
  //prepareZipx('song', encodeURIComponent(id), '')
  return;
	var url = url+'?id=' + encodeURIComponent(id);
    //showDialog(url);
    window.open(url);
};
//鎺ㄩ€佸棣栨瓕鏇�
function sendsongs(ids){
	var url = '/music/sendall';
	var id = getSelectValues(ids);
	if(id == ''){alert("娌℃湁璧勬簮鍙互鍙戦€侊紒");return;}
	var url = url+'?type=songs&id=' + encodeURIComponent(id);
	showDialog(url);
};

//鎺ㄨ崘
//32,姝屾洸锛�33锛屼笓杈戯紝34锛岃壓浜猴紝35锛屾瓕鍗曪紝36锛屾瓕鏇茶瘎璁猴紝37锛屼笓杈戣瘎璁猴紝38锛岃壓浜鸿瘎璁猴紝 39锛屾瓕鍗曡瘎璁猴紝 310锛� 灏忕粍璇濋锛� 311锛岀敤鎴凤紝 312锛屽皬缁�
function recommend(id,type,sid){
	var url = '/recommend/post';
	var url = url+'?object_id=' + encodeURIComponent(id)+"&type="+encodeURIComponent(type)+"&t="+1000*Math.random();
	if(sid) var url = url + '&sid='+ encodeURIComponent(sid);
	showDialog(url);
};

//鍐嶆帹
function retui(id){
	var url = '/recommend/feed/id/' + encodeURIComponent(id) ;
	showDialog(url);
};

//鍒嗕韩
function share(url){
	var url = '/feed/share/?url=' + encodeURIComponent(url);
	showDialog(url);
};

//鍙備笌姝屽崟
function addcollect(id){
	var url = '/collect/addcollect';
	var url = url+'?cid=' + encodeURIComponent(id)+"&"+Math.random();
	showDialog(url);
};

//涓撹緫鎯宠
function require(aid,type){
	var url = '/album/require';
	var url = url+'?id=' + encodeURIComponent(aid)+"&type="+encodeURIComponent(type);
	showDialog(url);
};

//浠嬬粛缁欏ソ鍙�
//33锛屼笓杈戯紝35锛屾瓕鍗�
function intro(id,type){
	var url = '/member/intro';
	var url = url+'?object_id=' + encodeURIComponent(id)+"&type="+encodeURIComponent(type);
	showDialog(url);
};

function friends(id){
	var url = '/member/myfriends/t/new/to_user_id/'+ encodeURIComponent(id);
	showDialog(url);
};

function attention(id,type){
	var url = '/member/attention/from/ajax/type/'+encodeURIComponent(type)+'/uid/'+ encodeURIComponent(id);
	showDialog(url);
};

function blacklist(uid){
	if(!confirm('Ta 灏嗕笉鑳�...\n- 鍏虫敞浣� (宸插叧娉ㄧ殑浼氳嚜鍔ㄥ彇娑堝叧娉�) \n- 缁欎綘鍙戠珯鍐呬俊\n- 缁欎綘鐣欒█锛屽洖澶嶄綘鐨勫垎浜瓑\n纭畾瑕佹妸 Ta 鍔犲叆榛戝悕鍗曞悧锛�')) return;
	window.location='/member/attention/uid/'+uid+'/type/3?_xiamitoken='+_xiamitoken;
}

//涓撹緫鏀惰棌鍒板皬缁�
function groupalbum(id){
	var url = '/group/pooladd/id/'+ encodeURIComponent(id);
	showDialog(url);
};

//鍔犲叆灏忕粍
function groupjoin(id){
	var url = '/group/join/id/'+ encodeURIComponent(id);
	showDialog(url);
};

//姝屽崟鏀惰棌鍒板皬缁�
function groupcollect(id){
	var url = '/group/pooladd/type/1/id/'+ encodeURIComponent(id);
	showDialog(url);
};

//淇敼涓汉浠嬬粛
function editprofile(id){
	var url = '/member/editprofile';
	var url = url+'?object_id=' + encodeURIComponent(id);
	showDialog(url);
};

function getshengxiao(yyyy){
    //by Go_Rush(闃胯垳) from http://ashun.cnblogs.com/
    var arr=['鐚�','楦�','鐙�','鐚�','榧�','鐗�','铏�','鍏�','榫�','铔�','椹�','缇�'];
    return /^\d{4}$/.test(yyyy)?arr[yyyy%12]:null;
};

// 鍙栨槦搴�, 鍙傛暟鍒嗗埆鏄� 鏈堜唤鍜屾棩鏈�
function getxingzuo(month,day){
    //by Go_Rush(闃胯垳) from http://ashun.cnblogs.com/
    var d=new Date(1999,month-1,day,0,0,0);
    var arr=[];
    arr.push(["榄旂警搴�",new Date(1999, 0, 1,0,0,0)])
    arr.push(["姘寸摱搴�",new Date(1999, 0,20,0,0,0)])
    arr.push(["鍙岄奔搴�",new Date(1999, 1,19,0,0,0)])
    arr.push(["鐗＄緤搴�",new Date(1999, 2,21,0,0,0)])
    arr.push(["閲戠墰搴�",new Date(1999, 3,21,0,0,0)])
    arr.push(["鍙屽瓙搴�",new Date(1999, 4,21,0,0,0)])
    arr.push(["宸ㄨ煿搴�",new Date(1999, 5,22,0,0,0)])
    arr.push(["鐙瓙搴�",new Date(1999, 6,23,0,0,0)])
    arr.push(["澶勫コ搴�",new Date(1999, 7,23,0,0,0)])
    arr.push(["澶╃Г搴�",new Date(1999, 8,23,0,0,0)])
    arr.push(["澶╄潕搴�",new Date(1999, 9,23,0,0,0)])
    arr.push(["灏勬墜搴�",new Date(1999,10,23,0,0,0)])
    arr.push(["榄旂警搴�",new Date(1999,11,22,0,0,0)])
    for(var i=arr.length-1;i>=0;i--){
        if (d>=arr[i][1]) return arr[i][0];
    }
};

/*
榄旂警搴�(12/22 - 1/19)銆佹按鐡跺骇(1/20 - 2/18)銆佸弻楸煎骇(2/19 - 3/20)銆佺墶缇婂骇(3/21 - 4/20)銆侀噾鐗涘骇(4/21 - 5/20)銆�
鍙屽瓙搴�(5/21 - 6/21)銆佸法锜瑰骇(6/22 - 7/22)銆佺嫯瀛愬骇(7/23 - 8/22)銆佸濂冲骇(8/23 - 9/22)銆佸ぉ绉ゅ骇(9/23 - 10/22)銆�
澶╄潕搴�(10/23 - 11/21)銆佸皠鎵嬪骇(11/22 - 12/21)
*/

function resendmail() {
	var url = '/member/regresend/type/ajax';
	showDialog(url);
};

//灏嗛€変腑姝屾洸鍒朵綔鎴愬鏇叉挱鏀緒idget
function makeMultiWidget(checkname){
	var ids = getSelectValues(checkname);
	if(ids == ''){alert("璇峰厛閫夋嫨姝屾洸!");return;}
	window.location='//emumo.xiami.com/widget/imulti?sid='+ids;
};

function makeMultiWidgetH(checkname){
	var ids = getValues(checkname);
	if(ids == ''){alert("璇峰厛閫夋嫨姝屾洸!");return;}
	window.location='//emumo.xiami.com/widget/imulti?sid='+ids;
};

//灏嗛€変腑鐨勬瓕鏇叉坊鍔犲埌姝屽崟
function collects(checkname){
	var ids = getSelectValues(checkname);
	if(ids == ''){alert("璇峰厛閫夋嫨姝屾洸!");return;}
	var url = '/song/collects/ids/'+ encodeURIComponent(ids);
	showDialog(url);
};

function collectsH(checkname){
	var ids = getValues(checkname);
	if(ids == ''){alert("璇峰厛閫夋嫨姝屾洸!");return;}
	var url = '/song/collects/ids/'+ encodeURIComponent(ids);
	showDialog(url);
};

//js妯℃澘瑙ｆ瀽 妯℃澘鍙橀噺绫讳技%uid% 浠�%鍙风晫瀹�
//str 涓洪渶瑙ｆ瀽鐨勬ā鏉跨殑html
//data涓簀son 鏁版嵁
//渚嬪浼犲叆str="<a href=/u/%uid%>%username%</a>"
//data={uid:1,username:'xiami'}
//鍒欒繑鍥� str = "<a href=/u/1>xiami</a>"
function parseTpl(str,data){
	var result;
	var patt = new RegExp("%([a-zA-z0-9]+)%");
	while ((result = patt.exec(str)) != null)  {
		var v = data[result[1]] || '';
		str = str.replace(new RegExp(result[0],"g"),v);
	}
	return str;
}
String.prototype.parseTpl=function(data){return parseTpl(this,data);};


//鏀惰棌鐨�
function player_collect(obj){
	//collect(obj.songId);
	//鎵�3鍒嗘敹钘忔垚濂借瘎鐨勬瓕鏇�
};

function player_collected(songId){
	thisMovie("musicPlayer").player_collected(songId);
}

//涓嬭浇
function player_download(obj){
	download(obj.songId);
};

//澶氶涓嬭浇
function player_downloadmulty(objAry){
	if(!objAry.length) {alert('璇烽€夋嫨闇€瑕佷笅杞界殑姝屾洸锛�');return;}
	var ids = '';
	for(var i=0;i<objAry.length;i++){
		if(ids != '') ids += ',';
		ids = ids + objAry[i].songId;
	}
	if(ids == ''){alert("璇烽€夋嫨闇€瑕佷笅杞界殑姝屾洸锛�");return;}
	var url = '//emumo.xiami.com/download/pay?id=' + encodeURIComponent(ids);
    //showDialog(url);
    window.open(url);
}

//娣诲姞
function player_add(obj){
	collect(obj.songId);
}

//鏇村
function player_more(objAry){
	var html = $('#tpl-gears_more').val().parseTpl(objAry);
	$('.gears_more').html(html).show();
	window.event.cancelBubble = true;
	return false;
}

//鍒嗕韩
function player_share(obj){
	share("//emumo.xiami.com/song/"+ obj.songId);
}


//鎺ㄨ崘
function player_recommend(obj){
	recommend(obj.songId,32);
};

//涓嶅枩娆�
function player_unlike(obj){

};

//璇勪环
function player_review(obj,num){
	num-=1;
	var url = '/song/review/id/'+ encodeURIComponent(obj.songId) + '/num/'+ encodeURIComponent(num);
	showDialog(url);
};

//姝屾洸鏈夐敊 姝岃瘝鏈夐敊 璇嶆洸涓嶅悓姝�
function player_reportlyric(obj,type){
	var url = '/song/reportlyric/type/'+type+'/id/'+ encodeURIComponent(obj.songId);
	showDialog(url);
};

//涓婁紶姝岃瘝
function player_uploadlyric(obj){
	var url = '/song/addlyric/id/'+ encodeURIComponent(obj.songId);
	showDialog(url);
};

//姝屾洸鏀瑰彉
function player_changeSong(obj){
	document.title= obj.songName + "鈥斺€旇櫨灏忕背鎵撶涓€︹€�";
};

//鎾斁瀹屾垚
function player_overSong(obj){
	//$.get("http://data.xiami.com/count/playrecord/sid/"+obj.songId+"?object_name="+obj.objectName+"&object_id="+obj.objectId);
	var target = "//emumo.xiami.com/count/playrecord/sid/"+obj.songId+"?object_name="+obj.objectName+"&object_id="+obj.objectId;
	$.ajax({type: "GET", url: target, dataType: "jsonp"});
};

function xiamiclick(obj,jl,lj,hx) {
	var wordTip = '';
	var href = '';
	wordTip = getTip(jl);

	if(lj) {//鏈夐摼鎺ョ殑鏃跺€欙紝闇€瑕佸湪閾炬帴鍚庨潰鍔犲叧閿瘝
		href = $(obj).attr("href");
		if(href.indexOf(wordTip)=='-1') {
			var join = '';
			if(hx) join = "?"; else join = "&";
			var new_href = $(obj).attr("href")+join+wordTip;
			$(obj).attr("href",new_href);
		}
		return true;
	}else {
		href = "/index/ajaxmemberooter?"+wordTip;
		$.post(href,{},function(txt) {
			$(obj).after(txt);
		});
		return true;
	}
};

function albumstore(id,type){
	var url = '/album/addstore/id/'+ encodeURIComponent(id)+'?type='+ encodeURIComponent(type);
	showDialog(url);
};

function getTip(num) {
	var Tip = '';
	switch(num) {
		//棣栭〉
		case 101:Tip = 'trace_index_welcome';break;
		case 102:Tip = 'trace_index_recommand';break;
		case 103:Tip = 'trace_index_navigator';break;
		case 104:Tip = 'trace_index_blog';break;
		case 105:Tip = 'trace_index_software';break;
		case 106:Tip = 'trace_index_xiami_fm';break;
		case 107:Tip = 'trace_index_event';break;
		case 108:Tip = 'trace_index_friendsfeed';break;
		case 109:Tip = 'trace_index_album_new';break;
		case 110:Tip = 'trace_index_album_updown';break;
		case 111:Tip = 'trace_index_album_want';break;
		case 112:Tip = 'trace_index_collect_sub';break;
		case 113:Tip = 'trace_index_collect_recommand';break;
		case 114:Tip = 'trace_index_ranking';break;
		case 115:Tip = 'trace_index_comment';break;
		case 116:Tip = 'trace_index_bottom';break;
		//闊充箰棰戦亾
		case 211:Tip = 'trace_music_navigator2';break;
		case 212:Tip = 'trace_music_filter';break;
		case 213:Tip = 'trace_music_recommand';break;
		case 214:Tip = 'trace_music_updown';break;
		case 222:Tip = 'trace_music_newalbum_list';break;
		case 221:Tip = 'trace_music_newalbum_category';break;
		case 231:Tip = 'trace_music_updown_list';break;
		case 241:Tip = 'trace_music_want_list';break;
		//姝屽崟
		case 311:Tip = 'trace_collect_orinew_navigator2';break;
		case 312:Tip = 'trace_collect_orinew_list';break;
		case 313:Tip = 'trace_collect_orinew_right';break;
		case 321:Tip = 'trace_collect_helpnew';break;
		case 331:Tip = 'trace_collect_sub_head';break;
		case 332:Tip = 'trace_collect_sub_list';break;
		case 333:Tip = 'trace_collect_sub_subpast';break;
		//灏忕粍
		case 411:Tip = 'trace_group_24hot';break;
		case 412:Tip = 'trace_group_update';break;
		case 413:Tip = 'trace_group_60min';break;
		case 414:Tip = 'trace_group_recommand';break;
		case 415:Tip = 'trace_group_manage';break;
		case 416:Tip = 'trace_group_myjoin';break;
		case 417:Tip = 'trace_group_friendjoin';break;
	}
	return Tip;
};

//鍠滄涓€浣嶈壓浜�
function artistLike(obj,url) {
	var id,load,uid;
	id = $(obj).attr("id");
	uid = $(obj).attr("rel");
	if(!uid) {
		if(confirm("鍛€锛佽繕鏈櫥褰曪紝鐜板湪瑕佺櫥褰曞悧锛�")) window.location = "https://login.xiami.com/member/login?done="+url;
		return ;
	}
	load = '<p><img width="16" height="16" alt="" src="//img.xiami.net/res/img/default/loading.gif"/></p>';
	$(obj).html(load);
	$.post('/artist/like',{ajax:1,likes:1,id:id, '_xiamitoken':_xiamitoken},function(data) {
		if(data==1) {
			var success = '<p>宸茶褰曪紱鏌ョ湅<a title="" href="/space/lib-artist/u/'+uid+'">鎴戝枩娆㈢殑鑹轰汉</a></p>';
		}
		$(obj).replaceWith(success);
	});
};

//涓撹緫棰勮
/**
*<div class="album_item100_thread" id="album_{$row.album_id}">
*<a class="preview" href="javascript:void(0)" onclick="album_preview(this)" id="{$row.album_id}" title="">棰勮</a>
*</div>
*鍦╠iv銆乤閲屽悇鍔爄d锛�
*/
function album_preview(obj) {
	var id,load,$preview,indexOf;
	id = $(obj).attr("id");
	$preview = $("#album_preview_"+id);
	$(obj).hide();
	indexOf = $(obj).attr("class").indexOf('current');
	$(".album_preview").hide();
	$("p .current").removeClass("current").html("棰勮");
	if($preview.html()) {
		if(indexOf=='-1') {
			$preview.show();
			$(obj).addClass("current").html("鍏抽棴棰勮").show();
		}else {
			$preview.hide();
			$(obj).removeClass("current").html("棰勮").show();
		}
	}else {
		load = '<div class="album_preview" id="album_preview_'+id+'"><img width="16" height="16" alt="" src="//img.xiami.net/res/img/default/loading.gif"/></div>';
		$("#album_"+id).after(load);
		$.post('/album/preview',{id:id},function(data) {
			$("#album_preview_"+id).html(data);
			$(obj).addClass("current").html("鍏抽棴棰勮").show();
		});
	}
};

//骞垮憡鏄剧ず
function display_ads(type,id){
	$.post('/search/sponsor',{type:type},function(data) {
		$("#"+id).html(data);
	});
};


//鎺ㄨ崘涓撻灞曠ず
function show_hot_events(type){
    $.post('/ajax/showevents',{type:type},function(data) {
        $('#hot_events_show').html(data);
        });
};

var $loading = $('<img class="load_feed loading" src="//img.xiami.net/res/img/default/loading2.gif" width="16" height="11" />');
var $loading2 = $('<img class="loading" src="//img.xiami.net/res/img/default/loading2.gif" width="16" height="11" />');

/*
function showFeedComment(feedId){
	var $feedWrap = $('#feed_item_'+feedId);
	var $comment=$feedWrap.find('.comments');
	if($comment.size()<1 && $feedWrap.find('.loading').size()<1){
		$feedWrap.append($loading);
		$.getJSON('/commentlist/feed',{id:feedId},function(data){
			if(data.status=='failed'){
				if(data.msg=='璇峰厛鐧诲綍'){
					var t = window.location;
					window.location='https://login.xiami.com/member/login?done='+t;
					return false;
				}
				alert(data.msg);return false;
			}
			$feedWrap.find('.loading').remove();
			$feedWrap.find('.feed_body').append(data.msg).find('.comments').hide().fadeIn();
			var $textarea = $feedWrap.find('.post textarea');
			var $count = $textarea.parent().next().find('.type_counts em');
			$textarea.focus().keyup(function(){
				var num = $textarea.val().length;
				if(num>140) alert('杈撳叆鍐呭璇烽檺鍒跺湪140瀛椾互鍐�');
				$count.html(num.toString());
			});
		});
	}else{
		if($comment.css('display')=='block') $comment.fadeOut();
		else $comment.fadeIn();
	}
};

function addFeedComment(feedId){
	var $feedWrap = $('#feed_item_'+feedId);
	var $form = $feedWrap.find('.post');
	var $ul = $form.prev().find('ul');
	var $textarea = $form.find('textarea');
	var $count = $textarea.parent().next().find('.type_counts em');
	var val = $textarea.val();
	if(val.length<3) {alert('璇疯緭鍏ヤ笉灏戜簬3涓瓧鐨勫唴瀹�');return false;}
	if(val.length>140) {alert('杈撳叆鍐呭璇烽檺鍒跺湪140瀛椾互鍐�');return false;}
	$feedWrap.find('.post .bt_sub2').hide().after($loading2);
	$.getJSON('/commentlist/add',{type:8,oid:feedId,content:val,mode:'ajax'},function(data){
		$feedWrap.find('.loading').remove();
		$feedWrap.find('.post .bt_sub2').fadeIn();
		if(data.status=='failed') {alert(data.errmsg);return;}
		$ul.append(data.output);
		$textarea.val('');
		var num = $textarea.val().length;
		$count.html(num.toString());
	});
};

function editFeedComment(commentId){
	var $commentItem = $('#feed_comment_item_'+commentId);
	var $editBox = '<li class="feed_comment_item"><p><textarea rows="2" cols="60"></textarea></p><p class="editbox_act"><input type="button" value="纭畾" class="bt_sub2"/> <input type="button" onclick="$(this).parent().parent().prev().fadeIn();$(this).parent().parent().remove();" value="鍙栨秷" class="bt_cancle2"/></p></li>';
	$commentItem.hide().after($editBox);
	$commentItem.next().find('textarea').val($commentItem.find('input').val()).focus();
	$commentItem.next().find('.bt_sub2').click(function(){
		var $btnSubmit = $(this);
		var $btnCancel = $btnSubmit.next();
		var $input = $btnSubmit.parent().prev().find('textarea');
		$btnSubmit.hide();
		$btnCancel.after($loading2).hide();
		$.getJSON('/commentlist/feededit',{content:$input.val(),id:commentId},function(data){
			if(data.status=='failed') {alert(data.errmsg);$commentItem.fadeIn().next().remove();return;}
			$commentItem.before(data.output).next().remove();
		});
	});
};

function delFeedComment(commentId){
	var $commentItem = $('#feed_comment_item_'+commentId);
	$commentItem.find('.feed_comment_item').append($loading2);
	$.post('/commentlist/del',{id:commentId,mode:'ajax'},function(data){
		$commentItem.fadeOut('fast',function(){$(this).remove();});
	});
};

function reFeedComment(commentId){
	var $commentItem = $('#feed_comment_item_'+commentId);
	if($commentItem.next().find('textarea').size()>0) return false;
	var $editBox = '<li class="feed_comment_reply"><p><textarea tabindex="4" rows="2" cols="60"></textarea></p><p class="editbox_act"><a  tabindex="6" href="javascript:;" onclick="$(this).parent().parent().prev().fadeIn();$(this).parent().parent().remove();">鍙栨秷</a> <input tabindex="5" type="button" value="纭畾" class="bt_sub2"/> </p></li>';
	$commentItem.after($editBox);
	$commentItem.next().find('textarea').val('鍥炲'+ $commentItem.find('.nickname').text() + ':').focus();
	$commentItem.next().find('.bt_sub2').click(function(){
		var $btnSubmit = $(this);
		var $btnCancel = $btnSubmit.next();
		var $input = $btnSubmit.parent().prev().find('textarea');
		$btnSubmit.hide();
		$btnCancel.after($loading2).hide();
		$.getJSON('/commentlist/feedre',{content:$input.val(),id:commentId,type:8},function(data){
			var $edbox = $commentItem.next();
			if(data.status=='failed') {alert(data.errmsg);$commentItem.fadeIn().next().remove();return;}
			$commentItem.after(data.output);
			$edbox.remove();
		});
	});
};
*/

function report(id,type){
	var url = '/ajax/report/id/'+ encodeURIComponent(id)+'/type/'+encodeURIComponent(type);
	showDialog(url);
};

$.fn.extend({getParent:function(level){
	var $this = $(this);
	for(var i=0;i<level;i++) $this = $this.parent();
	return $this;
}});

var __share_prefix = '#share_li_';
var __comment_div = '.share_comment';
var __comment_prefix = '#comment_li_';
var commentTpl = '<li id="comment_li_%_id%"><span>%msg%</span> - <a href="/u/%user_id%" class="nickname">%nick_name%</a> <span class="minor">( <a href="javascript:;" onclick="editShareComment(\'%shareId%\',\'%_id%\')">缂栬緫</a> | <a href="javascript:;" onclick="delShareComment(this,\'%shareId%\',\'%_id%\')">鍒犻櫎</a> )</span></li>';
var $editBox = '<li class="editbox"><p><textarea rows="2" cols="60"></textarea></p><p class="editbox_act"><input type="button" value="纭畾" class="bt_sub2"/> <input type="button" onclick="$(this).parent().parent().prev().fadeIn();$(this).parent().parent().remove();" value="鍙栨秷" class="bt_cancle2"/></p></li>';
var showShareComment = function(shareId){
	var $commentDiv = $(__share_prefix + shareId).find(__comment_div);
	$commentDiv.toggleClass('hidden');
	var $textarea = $commentDiv.find('.post textarea');
	var $count = $textarea.next().find('.type_counts em');
	$textarea.unbind().focus().keyup(function(e){
		var num = $textarea.val().length;
		if(num>140) $count.html('<span style="color:red">'+num.toString()+'</span>');
		else $count.html(num.toString());
	}).trigger('keyup');
};

var addShareComment = function(shareId){
	var $commentDiv = $(__share_prefix + shareId).find(__comment_div);
	var $submit = $commentDiv.find('.bt_sub2');
	var $textarea = $commentDiv.find('.post textarea');
	var val = $textarea.val();
	if(val.length<3) {alert('澶╁摢锛佷綘闅鹃亾涓嶄細鍐欑偣浠€涔堝悧锛熻繖涔堢煭锛燂紒锛�');return;}
	if(val.length>140){alert('OMG,浣犳€庝箞鏈夎繖涔堝搴熻瘽锛屼笉鏄嫹璐濈矘甯栫殑鍚�? 鏈€澶氬彧闇€瑕�140涓瓧鍝�:-)'); return;}
	$submit.hide().after($loading2);
	$.getJSON('/share/comment-add',{'msg':val,'shareId':shareId, '_xiamitoken':_xiamitoken},function(data){
		try{
			if(data.status=='failed'){$submit.show().next().remove();alert(data.msg);return;}
			if(data.status=='ok'){
				$textarea.val('').trigger('keyup');
				data.comment.shareId=shareId;
				$commentDiv.find('ul li:last').before(commentTpl.parseTpl(data.comment));
			}
		}catch(e){alert('鎻愪氦杩囩▼涓嚭鐜伴敊璇紒璇烽噸璇�');}
		$submit.show().next().remove();
	});
};


var editShareComment = function(shareId,commentId){
	var $commentDiv = $(__share_prefix + shareId).find(__comment_div);
	var $commentItem = $commentDiv.find(__comment_prefix +commentId);
	$commentItem.hide().after($editBox);
	$commentItem.next().find('textarea').focus().val($($commentItem.find('span')[0]).text());
	$commentItem.next().find('.bt_sub2').click(function(){
		var $btnSubmit = $(this);
		var $btnCancel = $btnSubmit.next();
		var $input = $btnSubmit.parent().prev().find('textarea');
		var val = $input.val();
		if(val.length<3) {alert('澶╁摢锛佷綘闅鹃亾涓嶄細鍐欑偣浠€涔堝悧锛熻繖涔堢煭锛燂紒锛�');return;}
		if(val.length>140){alert('OMG,浣犳€庝箞鏈夎繖涔堝搴熻瘽锛屼笉鏄嫹璐濈矘甯栫殑鍚�? 鏈€澶氬彧闇€瑕�140涓瓧鍝�:-)'); return;}
		$btnSubmit.hide();
		$btnCancel.after($loading2).hide();
		$.getJSON('/share/comment-edit',{msg:$input.val(),commentId:commentId,shareId:shareId, '_xiamitoken':_xiamitoken},function(data){
			if(data.status=='failed') {alert(data.msg);$commentItem.fadeIn().next().remove();return;}
			data.comment.shareId=shareId;
			$commentItem.before(commentTpl.parseTpl(data.comment)).next().remove();
			$commentItem.remove();
		});
	});
};

var delShare = function(_this,shareId){
	if(!confirm('纭畾瑕佸垹闄ゅ垎浜悧锛�')){return;}
	var $shareDiv = $(__share_prefix + shareId);
	$(_this).hide().after($loading2);
	$.getJSON('/share/del',{shareId:shareId, '_xiamitoken':_xiamitoken},function(data){
		if(data.status=='failed'){$(_this).show().next().remove();alert(data.msg);return;}
		if(data.status=='ok'){$shareDiv.fadeOut('fast',function(){$(this).remove();});}
		else{$(_this).show().next().remove();alert('鍒犻櫎澶辫触锛岃閲嶈瘯锛�');return;}
	});
}

var delShareComment = function(_this,shareId,commentId){
	if(!confirm('纭畾瑕佸垹闄よ瘎璁哄悧锛�')){return;}
	var $commentDiv = $(__share_prefix + shareId).find(__comment_div);
	var $commentItem = $commentDiv.find(__comment_prefix +commentId);
	$(_this).hide().after($loading2);
	$.getJSON('/share/comment-del',{commentId:commentId,shareId:shareId, '_xiamitoken':_xiamitoken},function(data){
		if(data.status=='failed'){$(_this).show().next().remove();alert(data.msg);return;}
		if(data.status=='ok'){$commentItem.fadeOut('fast',function(){$(this).remove();});}
		else{$(_this).show().next().remove();alert('鍒犻櫎澶辫触锛岃閲嶈瘯锛�');return;}
	});
}

var reShareComment = function(shareId,commentId){
	var $commentDiv = $(__share_prefix + shareId).find(__comment_div);
	var $commentItem = $commentDiv.find(__comment_prefix +commentId);
	if($commentItem.next().hasClass('editbox')) return;
	$commentItem.after($editBox);
	var $edbox = $commentItem.next();
	$edbox.find('textarea').focus().val('鍥炲'+ $commentItem.find('.nickname').text() + ': ');
	$edbox.find('.bt_sub2').click(function(){
		var $btnSubmit = $(this);
		var $btnCancel = $btnSubmit.next();
		var $input = $btnSubmit.parent().prev().find('textarea');
		var val = $input.val();
		if(val.length<3) {alert('澶╁摢锛佷綘闅鹃亾涓嶄細鍐欑偣浠€涔堝悧锛熻繖涔堢煭锛燂紒锛�');return;}
		if(val.length>140){alert('OMG,浣犳€庝箞鏈夎繖涔堝搴熻瘽锛屼笉鏄嫹璐濈矘甯栫殑鍚�? 鏈€澶氬彧闇€瑕�140涓瓧鍝�:-)'); return;}
		$btnSubmit.hide();
		$btnCancel.after($loading2).hide();
		$.getJSON('/share/comment-add',{'msg':$input.val(),'shareId':shareId,'commentId':commentId, '_xiamitoken':_xiamitoken},function(data){
			try{
				if(data.status=='failed'){alert(data.msg);$commentItem.fadeIn().next().remove();return;}
				if(data.status=='ok'){
					$edbox.remove();
					data.comment.shareId=shareId;
					$commentDiv.find('ul li:last').before(commentTpl.parseTpl(data.comment));
				}
			}catch(e){alert('鎻愪氦杩囩▼涓嚭鐜伴敊璇紒璇烽噸璇�');}
		});
	});
};

var parseUbbFlash = function(obj) {
	var self = $(obj).parent();
	var swf = self.data("swf");
	var WH = self.attr('rel');
	var height,width;
	if(WH) {
		var index = WH.indexOf('_');
		height=WH.substring(0,index);
		width=WH.substring(index+1);
	}else {
		width=600;height=450;
	}
	if(swf.indexOf("tudou.com") != -1){
		swf = swf.replace("v.swf","&autoPlay=true/v.swf")
	}

	if(swf.indexOf("yinyuetai.com") != -1){
		swf = swf.replace("v_0.swf","a_0.swf");
	}
	var tmp = '<object width="{width}" height="{height}" classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" codebase="http://fpdownload.macromedia.com/pub/shockwave/cabs/flash/swflash.cab#version=8,0,0,0"><param name="allowScriptAccess" value="never"><flashvars value="playMovie=true&auto=1&adss=0&isAutoPlay=true" /><param name="autoplay" value="true"><param name="allowFullScreen" value="false"><param name="movie" value="{swf}"><param value="opaque" name="wmode"><embed width="{width}" height="{height}" src="{swf}" quality="high" wmode="Opaque" allowscriptaccess="never" autoplay="true" flashvars="playMovie=true&auto=1&adss=0&isAutoPlay=true" type="application/x-shockwave-flash"></object>';

	var html = tmp.replace(/\{swf\}/g,swf);
	html = html.replace(/\{width\}/g,width);
	html = html.replace(/\{height\}/g,height);
	self.html(html);
	$(this).unbind("click");
	return false;
};

var reqStat = function(name) {
	$.get("//emumo.xiami.com/statclick/req/"+name);
};

var strUnescape = function(str) {
	var exps = {'&amp;':'&', '&quot;':'"', '&#39;':"'", '&lt;':'<', '&gt;':'>'};
	for (var re in exps) {
		str = str.split(re).join(exps[re]);
	}
	return str;
};