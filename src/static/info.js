function b64(e){var t="";var n=new Uint8Array(e);var r=n.byteLength;for(var i=0;i<r;i++){t+=String.fromCharCode(n[i])}return window.btoa(t)}

$(document).ready(function () {
    // var socket = io.connect('http://127.0.0.1:5000');
    var successOptions = {
        autoHideDelay: 20000,
        showAnimation: "fadeIn",
        hideAnimation: "fadeOut",
        hideDuration: 700,
        arrowShow: false,
        className: "success",
    };
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/info');
    socket.on('connect', function () {
        socket.emit('my_event', 'User has connected!');
    });
    socket.on('my_response', function (msg) {
        // $("#name").append('<li>' + msg + '</li>');
        $("#name").html('<b>' + msg.name + '</b>');
        $("#mssv").html('<b>' + msg.mssv + '</b>');
        // console.log(typeof msg);
        var table_name = "";
        var table_mssv = "";
        var table = "";
        for (i in msg) {
            index = msg[i]
            console.log()
            table_name = '<table><tr><td rowspan="2"><img src="data:image/png;base64,'+ index.face_crop +'"'+'width="120" height="120"></td><td><b>' + index.name + '</b></td></tr><tr><td id="mssv"><b>'+index.mssv+'</b></td></tr></table>'
            table = table + '<div class="row justify-content-md-center">' + table_name+ '</div>';
        }
        $("#show_info").html(table)
    });
});