var Bc=Object.defineProperty;var zc=(i,t,e)=>t in i?Bc(i,t,{enumerable:!0,configurable:!0,writable:!0,value:e}):i[t]=e;var kt=(i,t,e)=>zc(i,typeof t!="symbol"?t+"":t,e);import"../chunks/Bzak7iHL.js";import{o as jl,a as Zl}from"../chunks/GG5zm9kr.js";import{s as me,c as va,h as zt,g as B,p as ys,aB as kc,a as Es,d as yt,e as bt,n as Hc,r as xt,t as Ke,u as Gn,f as Kl,j as Vc}from"../chunks/CpWkWWOo.js";import{s as fe,d as $l,a as Fe}from"../chunks/BlVfL1ME.js";import{i as kn}from"../chunks/B4yTwGkE.js";import{e as _s,i as hr}from"../chunks/CGEBXrjl.js";import{a as _e,f as Se,c as Gc}from"../chunks/CHOnp4oo.js";import{s as ve,r as xa}from"../chunks/A7po6GxK.js";import{s as Us}from"../chunks/aVbAZ-t7.js";import{s as Sr}from"../chunks/Cx-f-Pzo.js";import{b as Ma}from"../chunks/sZcqyNBA.js";import{b as Jl}from"../chunks/BnXDGOmJ.js";import{s as Wc,a as Xc}from"../chunks/C6HuKgyx.js";import{b as Do}from"../chunks/g4XF1_b9.js";import{b as Yc}from"../chunks/CJsMJEun.js";import{p as vs}from"../chunks/V6gjw5Ec.js";import{N as Sa}from"../chunks/CcUbQ_Wl.js";import{i as qc}from"../chunks/BUoSzNdg.js";import{a as gi}from"../chunks/B7CfdQuM.js";import{e as jc}from"../chunks/MAY1QfFZ.js";/**
 * @license
 * Copyright 2010-2024 Three.js Authors
 * SPDX-License-Identifier: MIT
 */const vo="172",Hi={ROTATE:0,DOLLY:1,PAN:2},Oi={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},Zc=0,Lo=1,Kc=2,Ql=1,$c=2,An=3,Yn=0,Xe=1,gn=2,Cn=0,Vi=1,Le=2,Uo=3,Io=4,Jc=5,ii=100,Qc=101,th=102,eh=103,nh=104,ih=200,sh=201,rh=202,ah=203,ya=204,Ea=205,oh=206,lh=207,ch=208,hh=209,uh=210,dh=211,fh=212,ph=213,mh=214,ba=0,Ta=1,wa=2,qi=3,Aa=4,Ra=5,Ca=6,Pa=7,tc=0,gh=1,_h=2,Wn=0,vh=1,xh=2,Mh=3,ec=4,Sh=5,yh=6,Eh=7,nc=300,ji=301,Zi=302,Da=303,La=304,Pr=306,Ua=1e3,ri=1001,Ia=1002,Je=1003,bh=1004,Is=1005,vn=1006,Br=1007,ai=1008,Ln=1009,ic=1010,sc=1011,Ms=1012,xo=1013,li=1014,xn=1015,Pn=1016,Mo=1017,So=1018,Ki=1020,rc=35902,ac=1021,oc=1022,dn=1023,lc=1024,cc=1025,Gi=1026,$i=1027,yo=1028,Eo=1029,hc=1030,bo=1031,To=1033,ur=33776,dr=33777,fr=33778,pr=33779,Na=35840,Fa=35841,Oa=35842,Ba=35843,za=36196,ka=37492,Ha=37496,Va=37808,Ga=37809,Wa=37810,Xa=37811,Ya=37812,qa=37813,ja=37814,Za=37815,Ka=37816,$a=37817,Ja=37818,Qa=37819,to=37820,eo=37821,mr=36492,no=36494,io=36495,uc=36283,so=36284,ro=36285,ao=36286,Th=3200,wh=3201,dc=0,Ah=1,Vn="",an="srgb",Ji="srgb-linear",yr="linear",se="srgb",_i=7680,No=519,Rh=512,Ch=513,Ph=514,fc=515,Dh=516,Lh=517,Uh=518,Ih=519,oo=35044,Fo="300 es",Rn=2e3,Er=2001;class hi{addEventListener(t,e){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[t]===void 0&&(n[t]=[]),n[t].indexOf(e)===-1&&n[t].push(e)}hasEventListener(t,e){if(this._listeners===void 0)return!1;const n=this._listeners;return n[t]!==void 0&&n[t].indexOf(e)!==-1}removeEventListener(t,e){if(this._listeners===void 0)return;const s=this._listeners[t];if(s!==void 0){const r=s.indexOf(e);r!==-1&&s.splice(r,1)}}dispatchEvent(t){if(this._listeners===void 0)return;const n=this._listeners[t.type];if(n!==void 0){t.target=this;const s=n.slice(0);for(let r=0,a=s.length;r<a;r++)s[r].call(this,t);t.target=null}}}const Ie=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],gr=Math.PI/180,lo=180/Math.PI;function Xn(){const i=Math.random()*4294967295|0,t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(Ie[i&255]+Ie[i>>8&255]+Ie[i>>16&255]+Ie[i>>24&255]+"-"+Ie[t&255]+Ie[t>>8&255]+"-"+Ie[t>>16&15|64]+Ie[t>>24&255]+"-"+Ie[e&63|128]+Ie[e>>8&255]+"-"+Ie[e>>16&255]+Ie[e>>24&255]+Ie[n&255]+Ie[n>>8&255]+Ie[n>>16&255]+Ie[n>>24&255]).toLowerCase()}function Yt(i,t,e){return Math.max(t,Math.min(e,i))}function Nh(i,t){return(i%t+t)%t}function zr(i,t,e){return(1-e)*i+e*t}function _n(i,t){switch(t.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function re(i,t){switch(t.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const Fh={DEG2RAD:gr};class St{constructor(t=0,e=0){St.prototype.isVector2=!0,this.x=t,this.y=e}get width(){return this.x}set width(t){this.x=t}get height(){return this.y}set height(t){this.y=t}set(t,e){return this.x=t,this.y=e,this}setScalar(t){return this.x=t,this.y=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y)}copy(t){return this.x=t.x,this.y=t.y,this}add(t){return this.x+=t.x,this.y+=t.y,this}addScalar(t){return this.x+=t,this.y+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this}subScalar(t){return this.x-=t,this.y-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this}multiply(t){return this.x*=t.x,this.y*=t.y,this}multiplyScalar(t){return this.x*=t,this.y*=t,this}divide(t){return this.x/=t.x,this.y/=t.y,this}divideScalar(t){return this.multiplyScalar(1/t)}applyMatrix3(t){const e=this.x,n=this.y,s=t.elements;return this.x=s[0]*e+s[3]*n+s[6],this.y=s[1]*e+s[4]*n+s[7],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this}clamp(t,e){return this.x=Yt(this.x,t.x,e.x),this.y=Yt(this.y,t.y,e.y),this}clampScalar(t,e){return this.x=Yt(this.x,t,e),this.y=Yt(this.y,t,e),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Yt(n,t,e))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(t){return this.x*t.x+this.y*t.y}cross(t){return this.x*t.y-this.y*t.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(t){const e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;const n=this.dot(t)/e;return Math.acos(Yt(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){const e=this.x-t.x,n=this.y-t.y;return e*e+n*n}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this}equals(t){return t.x===this.x&&t.y===this.y}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this}rotateAround(t,e){const n=Math.cos(e),s=Math.sin(e),r=this.x-t.x,a=this.y-t.y;return this.x=r*n-a*s+t.x,this.y=r*s+a*n+t.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class Ht{constructor(t,e,n,s,r,a,o,l,c){Ht.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],t!==void 0&&this.set(t,e,n,s,r,a,o,l,c)}set(t,e,n,s,r,a,o,l,c){const h=this.elements;return h[0]=t,h[1]=s,h[2]=o,h[3]=e,h[4]=r,h[5]=l,h[6]=n,h[7]=a,h[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(t){const e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],this}extractBasis(t,e,n){return t.setFromMatrix3Column(this,0),e.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(t){const e=t.elements;return this.set(e[0],e[4],e[8],e[1],e[5],e[9],e[2],e[6],e[10]),this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){const n=t.elements,s=e.elements,r=this.elements,a=n[0],o=n[3],l=n[6],c=n[1],h=n[4],d=n[7],p=n[2],u=n[5],g=n[8],_=s[0],m=s[3],f=s[6],T=s[1],E=s[4],y=s[7],D=s[2],w=s[5],C=s[8];return r[0]=a*_+o*T+l*D,r[3]=a*m+o*E+l*w,r[6]=a*f+o*y+l*C,r[1]=c*_+h*T+d*D,r[4]=c*m+h*E+d*w,r[7]=c*f+h*y+d*C,r[2]=p*_+u*T+g*D,r[5]=p*m+u*E+g*w,r[8]=p*f+u*y+g*C,this}multiplyScalar(t){const e=this.elements;return e[0]*=t,e[3]*=t,e[6]*=t,e[1]*=t,e[4]*=t,e[7]*=t,e[2]*=t,e[5]*=t,e[8]*=t,this}determinant(){const t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],l=t[6],c=t[7],h=t[8];return e*a*h-e*o*c-n*r*h+n*o*l+s*r*c-s*a*l}invert(){const t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],l=t[6],c=t[7],h=t[8],d=h*a-o*c,p=o*l-h*r,u=c*r-a*l,g=e*d+n*p+s*u;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);const _=1/g;return t[0]=d*_,t[1]=(s*c-h*n)*_,t[2]=(o*n-s*a)*_,t[3]=p*_,t[4]=(h*e-s*l)*_,t[5]=(s*r-o*e)*_,t[6]=u*_,t[7]=(n*l-c*e)*_,t[8]=(a*e-n*r)*_,this}transpose(){let t;const e=this.elements;return t=e[1],e[1]=e[3],e[3]=t,t=e[2],e[2]=e[6],e[6]=t,t=e[5],e[5]=e[7],e[7]=t,this}getNormalMatrix(t){return this.setFromMatrix4(t).invert().transpose()}transposeIntoArray(t){const e=this.elements;return t[0]=e[0],t[1]=e[3],t[2]=e[6],t[3]=e[1],t[4]=e[4],t[5]=e[7],t[6]=e[2],t[7]=e[5],t[8]=e[8],this}setUvTransform(t,e,n,s,r,a,o){const l=Math.cos(r),c=Math.sin(r);return this.set(n*l,n*c,-n*(l*a+c*o)+a+t,-s*c,s*l,-s*(-c*a+l*o)+o+e,0,0,1),this}scale(t,e){return this.premultiply(kr.makeScale(t,e)),this}rotate(t){return this.premultiply(kr.makeRotation(-t)),this}translate(t,e){return this.premultiply(kr.makeTranslation(t,e)),this}makeTranslation(t,e){return t.isVector2?this.set(1,0,t.x,0,1,t.y,0,0,1):this.set(1,0,t,0,1,e,0,0,1),this}makeRotation(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,n,e,0,0,0,1),this}makeScale(t,e){return this.set(t,0,0,0,e,0,0,0,1),this}equals(t){const e=this.elements,n=t.elements;for(let s=0;s<9;s++)if(e[s]!==n[s])return!1;return!0}fromArray(t,e=0){for(let n=0;n<9;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){const n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t}clone(){return new this.constructor().fromArray(this.elements)}}const kr=new Ht;function pc(i){for(let t=i.length-1;t>=0;--t)if(i[t]>=65535)return!0;return!1}function br(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function Oh(){const i=br("canvas");return i.style.display="block",i}const Oo={};function Fi(i){i in Oo||(Oo[i]=!0,console.warn(i))}function Bh(i,t,e){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(t,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,e);break;default:n()}}setTimeout(r,e)})}function zh(i){const t=i.elements;t[2]=.5*t[2]+.5*t[3],t[6]=.5*t[6]+.5*t[7],t[10]=.5*t[10]+.5*t[11],t[14]=.5*t[14]+.5*t[15]}function kh(i){const t=i.elements;t[11]===-1?(t[10]=-t[10]-1,t[14]=-t[14]):(t[10]=-t[10],t[14]=-t[14]+1)}const Bo=new Ht().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),zo=new Ht().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function Hh(){const i={enabled:!0,workingColorSpace:Ji,spaces:{},convert:function(s,r,a){return this.enabled===!1||r===a||!r||!a||(this.spaces[r].transfer===se&&(s.r=Dn(s.r),s.g=Dn(s.g),s.b=Dn(s.b)),this.spaces[r].primaries!==this.spaces[a].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===se&&(s.r=Wi(s.r),s.g=Wi(s.g),s.b=Wi(s.b))),s},fromWorkingColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},toWorkingColorSpace:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===Vn?yr:this.spaces[s].transfer},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,a){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[a].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace}},t=[.64,.33,.3,.6,.15,.06],e=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[Ji]:{primaries:t,whitePoint:n,transfer:yr,toXYZ:Bo,fromXYZ:zo,luminanceCoefficients:e,workingColorSpaceConfig:{unpackColorSpace:an},outputColorSpaceConfig:{drawingBufferColorSpace:an}},[an]:{primaries:t,whitePoint:n,transfer:se,toXYZ:Bo,fromXYZ:zo,luminanceCoefficients:e,outputColorSpaceConfig:{drawingBufferColorSpace:an}}}),i}const Qt=Hh();function Dn(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function Wi(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let vi;class Vh{static getDataURL(t){if(/^data:/i.test(t.src)||typeof HTMLCanvasElement>"u")return t.src;let e;if(t instanceof HTMLCanvasElement)e=t;else{vi===void 0&&(vi=br("canvas")),vi.width=t.width,vi.height=t.height;const n=vi.getContext("2d");t instanceof ImageData?n.putImageData(t,0,0):n.drawImage(t,0,0,t.width,t.height),e=vi}return e.width>2048||e.height>2048?(console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons",t),e.toDataURL("image/jpeg",.6)):e.toDataURL("image/png")}static sRGBToLinear(t){if(typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap){const e=br("canvas");e.width=t.width,e.height=t.height;const n=e.getContext("2d");n.drawImage(t,0,0,t.width,t.height);const s=n.getImageData(0,0,t.width,t.height),r=s.data;for(let a=0;a<r.length;a++)r[a]=Dn(r[a]/255)*255;return n.putImageData(s,0,0),e}else if(t.data){const e=t.data.slice(0);for(let n=0;n<e.length;n++)e instanceof Uint8Array||e instanceof Uint8ClampedArray?e[n]=Math.floor(Dn(e[n]/255)*255):e[n]=Dn(e[n]);return{data:e,width:t.width,height:t.height}}else return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),t}}let Gh=0;class mc{constructor(t=null){this.isSource=!0,Object.defineProperty(this,"id",{value:Gh++}),this.uuid=Xn(),this.data=t,this.dataReady=!0,this.version=0}set needsUpdate(t){t===!0&&this.version++}toJSON(t){const e=t===void 0||typeof t=="string";if(!e&&t.images[this.uuid]!==void 0)return t.images[this.uuid];const n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let a=0,o=s.length;a<o;a++)s[a].isDataTexture?r.push(Hr(s[a].image)):r.push(Hr(s[a]))}else r=Hr(s);n.url=r}return e||(t.images[this.uuid]=n),n}}function Hr(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?Vh.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(console.warn("THREE.Texture: Unable to serialize Texture."),{})}let Wh=0;class Pe extends hi{constructor(t=Pe.DEFAULT_IMAGE,e=Pe.DEFAULT_MAPPING,n=ri,s=ri,r=vn,a=ai,o=dn,l=Ln,c=Pe.DEFAULT_ANISOTROPY,h=Vn){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:Wh++}),this.uuid=Xn(),this.name="",this.source=new mc(t),this.mipmaps=[],this.mapping=e,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=a,this.anisotropy=c,this.format=o,this.internalFormat=null,this.type=l,this.offset=new St(0,0),this.repeat=new St(1,1),this.center=new St(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Ht,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=h,this.userData={},this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.pmremVersion=0}get image(){return this.source.data}set image(t=null){this.source.data=t}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}clone(){return new this.constructor().copy(this)}copy(t){return this.name=t.name,this.source=t.source,this.mipmaps=t.mipmaps.slice(0),this.mapping=t.mapping,this.channel=t.channel,this.wrapS=t.wrapS,this.wrapT=t.wrapT,this.magFilter=t.magFilter,this.minFilter=t.minFilter,this.anisotropy=t.anisotropy,this.format=t.format,this.internalFormat=t.internalFormat,this.type=t.type,this.offset.copy(t.offset),this.repeat.copy(t.repeat),this.center.copy(t.center),this.rotation=t.rotation,this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrix.copy(t.matrix),this.generateMipmaps=t.generateMipmaps,this.premultiplyAlpha=t.premultiplyAlpha,this.flipY=t.flipY,this.unpackAlignment=t.unpackAlignment,this.colorSpace=t.colorSpace,this.renderTarget=t.renderTarget,this.isRenderTargetTexture=t.isRenderTargetTexture,this.userData=JSON.parse(JSON.stringify(t.userData)),this.needsUpdate=!0,this}toJSON(t){const e=t===void 0||typeof t=="string";if(!e&&t.textures[this.uuid]!==void 0)return t.textures[this.uuid];const n={metadata:{version:4.6,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(t).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),e||(t.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(t){if(this.mapping!==nc)return t;if(t.applyMatrix3(this.matrix),t.x<0||t.x>1)switch(this.wrapS){case Ua:t.x=t.x-Math.floor(t.x);break;case ri:t.x=t.x<0?0:1;break;case Ia:Math.abs(Math.floor(t.x)%2)===1?t.x=Math.ceil(t.x)-t.x:t.x=t.x-Math.floor(t.x);break}if(t.y<0||t.y>1)switch(this.wrapT){case Ua:t.y=t.y-Math.floor(t.y);break;case ri:t.y=t.y<0?0:1;break;case Ia:Math.abs(Math.floor(t.y)%2)===1?t.y=Math.ceil(t.y)-t.y:t.y=t.y-Math.floor(t.y);break}return this.flipY&&(t.y=1-t.y),t}set needsUpdate(t){t===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(t){t===!0&&this.pmremVersion++}}Pe.DEFAULT_IMAGE=null;Pe.DEFAULT_MAPPING=nc;Pe.DEFAULT_ANISOTROPY=1;class oe{constructor(t=0,e=0,n=0,s=1){oe.prototype.isVector4=!0,this.x=t,this.y=e,this.z=n,this.w=s}get width(){return this.z}set width(t){this.z=t}get height(){return this.w}set height(t){this.w=t}set(t,e,n,s){return this.x=t,this.y=e,this.z=n,this.w=s,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this.w=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setW(t){return this.w=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;case 3:this.w=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this.w=t.w!==void 0?t.w:1,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this.w+=t.w,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this.w+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this.w=t.w+e.w,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this.w+=t.w*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this.w-=t.w,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this.w-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this.w=t.w-e.w,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this.w*=t.w,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this.w*=t,this}applyMatrix4(t){const e=this.x,n=this.y,s=this.z,r=this.w,a=t.elements;return this.x=a[0]*e+a[4]*n+a[8]*s+a[12]*r,this.y=a[1]*e+a[5]*n+a[9]*s+a[13]*r,this.z=a[2]*e+a[6]*n+a[10]*s+a[14]*r,this.w=a[3]*e+a[7]*n+a[11]*s+a[15]*r,this}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this.w/=t.w,this}divideScalar(t){return this.multiplyScalar(1/t)}setAxisAngleFromQuaternion(t){this.w=2*Math.acos(t.w);const e=Math.sqrt(1-t.w*t.w);return e<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=t.x/e,this.y=t.y/e,this.z=t.z/e),this}setAxisAngleFromRotationMatrix(t){let e,n,s,r;const l=t.elements,c=l[0],h=l[4],d=l[8],p=l[1],u=l[5],g=l[9],_=l[2],m=l[6],f=l[10];if(Math.abs(h-p)<.01&&Math.abs(d-_)<.01&&Math.abs(g-m)<.01){if(Math.abs(h+p)<.1&&Math.abs(d+_)<.1&&Math.abs(g+m)<.1&&Math.abs(c+u+f-3)<.1)return this.set(1,0,0,0),this;e=Math.PI;const E=(c+1)/2,y=(u+1)/2,D=(f+1)/2,w=(h+p)/4,C=(d+_)/4,I=(g+m)/4;return E>y&&E>D?E<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(E),s=w/n,r=C/n):y>D?y<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(y),n=w/s,r=I/s):D<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(D),n=C/r,s=I/r),this.set(n,s,r,e),this}let T=Math.sqrt((m-g)*(m-g)+(d-_)*(d-_)+(p-h)*(p-h));return Math.abs(T)<.001&&(T=1),this.x=(m-g)/T,this.y=(d-_)/T,this.z=(p-h)/T,this.w=Math.acos((c+u+f-1)/2),this}setFromMatrixPosition(t){const e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this.w=e[15],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this.w=Math.min(this.w,t.w),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this.w=Math.max(this.w,t.w),this}clamp(t,e){return this.x=Yt(this.x,t.x,e.x),this.y=Yt(this.y,t.y,e.y),this.z=Yt(this.z,t.z,e.z),this.w=Yt(this.w,t.w,e.w),this}clampScalar(t,e){return this.x=Yt(this.x,t,e),this.y=Yt(this.y,t,e),this.z=Yt(this.z,t,e),this.w=Yt(this.w,t,e),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Yt(n,t,e))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z+this.w*t.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this.w+=(t.w-this.w)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this.w=t.w+(e.w-t.w)*n,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z&&t.w===this.w}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this.w=t[e+3],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t[e+3]=this.w,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this.w=t.getW(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class Xh extends hi{constructor(t=1,e=1,n={}){super(),this.isRenderTarget=!0,this.width=t,this.height=e,this.depth=1,this.scissor=new oe(0,0,t,e),this.scissorTest=!1,this.viewport=new oe(0,0,t,e);const s={width:t,height:e,depth:1};n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:vn,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1},n);const r=new Pe(s,n.mapping,n.wrapS,n.wrapT,n.magFilter,n.minFilter,n.format,n.type,n.anisotropy,n.colorSpace);r.flipY=!1,r.generateMipmaps=n.generateMipmaps,r.internalFormat=n.internalFormat,this.textures=[];const a=n.count;for(let o=0;o<a;o++)this.textures[o]=r.clone(),this.textures[o].isRenderTargetTexture=!0,this.textures[o].renderTarget=this;this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples}get texture(){return this.textures[0]}set texture(t){this.textures[0]=t}set depthTexture(t){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),t!==null&&(t.renderTarget=this),this._depthTexture=t}get depthTexture(){return this._depthTexture}setSize(t,e,n=1){if(this.width!==t||this.height!==e||this.depth!==n){this.width=t,this.height=e,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=t,this.textures[s].image.height=e,this.textures[s].image.depth=n;this.dispose()}this.viewport.set(0,0,t,e),this.scissor.set(0,0,t,e)}clone(){return new this.constructor().copy(this)}copy(t){this.width=t.width,this.height=t.height,this.depth=t.depth,this.scissor.copy(t.scissor),this.scissorTest=t.scissorTest,this.viewport.copy(t.viewport),this.textures.length=0;for(let n=0,s=t.textures.length;n<s;n++)this.textures[n]=t.textures[n].clone(),this.textures[n].isRenderTargetTexture=!0,this.textures[n].renderTarget=this;const e=Object.assign({},t.texture.image);return this.texture.source=new mc(e),this.depthBuffer=t.depthBuffer,this.stencilBuffer=t.stencilBuffer,this.resolveDepthBuffer=t.resolveDepthBuffer,this.resolveStencilBuffer=t.resolveStencilBuffer,t.depthTexture!==null&&(this.depthTexture=t.depthTexture.clone()),this.samples=t.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class fn extends Xh{constructor(t=1,e=1,n={}){super(t,e,n),this.isWebGLRenderTarget=!0}}class gc extends Pe{constructor(t=null,e=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:t,width:e,height:n,depth:s},this.magFilter=Je,this.minFilter=Je,this.wrapR=ri,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(t){this.layerUpdates.add(t)}clearLayerUpdates(){this.layerUpdates.clear()}}class Yh extends Pe{constructor(t=null,e=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:t,width:e,height:n,depth:s},this.magFilter=Je,this.minFilter=Je,this.wrapR=ri,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class ci{constructor(t=0,e=0,n=0,s=1){this.isQuaternion=!0,this._x=t,this._y=e,this._z=n,this._w=s}static slerpFlat(t,e,n,s,r,a,o){let l=n[s+0],c=n[s+1],h=n[s+2],d=n[s+3];const p=r[a+0],u=r[a+1],g=r[a+2],_=r[a+3];if(o===0){t[e+0]=l,t[e+1]=c,t[e+2]=h,t[e+3]=d;return}if(o===1){t[e+0]=p,t[e+1]=u,t[e+2]=g,t[e+3]=_;return}if(d!==_||l!==p||c!==u||h!==g){let m=1-o;const f=l*p+c*u+h*g+d*_,T=f>=0?1:-1,E=1-f*f;if(E>Number.EPSILON){const D=Math.sqrt(E),w=Math.atan2(D,f*T);m=Math.sin(m*w)/D,o=Math.sin(o*w)/D}const y=o*T;if(l=l*m+p*y,c=c*m+u*y,h=h*m+g*y,d=d*m+_*y,m===1-o){const D=1/Math.sqrt(l*l+c*c+h*h+d*d);l*=D,c*=D,h*=D,d*=D}}t[e]=l,t[e+1]=c,t[e+2]=h,t[e+3]=d}static multiplyQuaternionsFlat(t,e,n,s,r,a){const o=n[s],l=n[s+1],c=n[s+2],h=n[s+3],d=r[a],p=r[a+1],u=r[a+2],g=r[a+3];return t[e]=o*g+h*d+l*u-c*p,t[e+1]=l*g+h*p+c*d-o*u,t[e+2]=c*g+h*u+o*p-l*d,t[e+3]=h*g-o*d-l*p-c*u,t}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get w(){return this._w}set w(t){this._w=t,this._onChangeCallback()}set(t,e,n,s){return this._x=t,this._y=e,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(t){return this._x=t.x,this._y=t.y,this._z=t.z,this._w=t.w,this._onChangeCallback(),this}setFromEuler(t,e=!0){const n=t._x,s=t._y,r=t._z,a=t._order,o=Math.cos,l=Math.sin,c=o(n/2),h=o(s/2),d=o(r/2),p=l(n/2),u=l(s/2),g=l(r/2);switch(a){case"XYZ":this._x=p*h*d+c*u*g,this._y=c*u*d-p*h*g,this._z=c*h*g+p*u*d,this._w=c*h*d-p*u*g;break;case"YXZ":this._x=p*h*d+c*u*g,this._y=c*u*d-p*h*g,this._z=c*h*g-p*u*d,this._w=c*h*d+p*u*g;break;case"ZXY":this._x=p*h*d-c*u*g,this._y=c*u*d+p*h*g,this._z=c*h*g+p*u*d,this._w=c*h*d-p*u*g;break;case"ZYX":this._x=p*h*d-c*u*g,this._y=c*u*d+p*h*g,this._z=c*h*g-p*u*d,this._w=c*h*d+p*u*g;break;case"YZX":this._x=p*h*d+c*u*g,this._y=c*u*d+p*h*g,this._z=c*h*g-p*u*d,this._w=c*h*d-p*u*g;break;case"XZY":this._x=p*h*d-c*u*g,this._y=c*u*d-p*h*g,this._z=c*h*g+p*u*d,this._w=c*h*d+p*u*g;break;default:console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: "+a)}return e===!0&&this._onChangeCallback(),this}setFromAxisAngle(t,e){const n=e/2,s=Math.sin(n);return this._x=t.x*s,this._y=t.y*s,this._z=t.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(t){const e=t.elements,n=e[0],s=e[4],r=e[8],a=e[1],o=e[5],l=e[9],c=e[2],h=e[6],d=e[10],p=n+o+d;if(p>0){const u=.5/Math.sqrt(p+1);this._w=.25/u,this._x=(h-l)*u,this._y=(r-c)*u,this._z=(a-s)*u}else if(n>o&&n>d){const u=2*Math.sqrt(1+n-o-d);this._w=(h-l)/u,this._x=.25*u,this._y=(s+a)/u,this._z=(r+c)/u}else if(o>d){const u=2*Math.sqrt(1+o-n-d);this._w=(r-c)/u,this._x=(s+a)/u,this._y=.25*u,this._z=(l+h)/u}else{const u=2*Math.sqrt(1+d-n-o);this._w=(a-s)/u,this._x=(r+c)/u,this._y=(l+h)/u,this._z=.25*u}return this._onChangeCallback(),this}setFromUnitVectors(t,e){let n=t.dot(e)+1;return n<Number.EPSILON?(n=0,Math.abs(t.x)>Math.abs(t.z)?(this._x=-t.y,this._y=t.x,this._z=0,this._w=n):(this._x=0,this._y=-t.z,this._z=t.y,this._w=n)):(this._x=t.y*e.z-t.z*e.y,this._y=t.z*e.x-t.x*e.z,this._z=t.x*e.y-t.y*e.x,this._w=n),this.normalize()}angleTo(t){return 2*Math.acos(Math.abs(Yt(this.dot(t),-1,1)))}rotateTowards(t,e){const n=this.angleTo(t);if(n===0)return this;const s=Math.min(1,e/n);return this.slerp(t,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(t){return this._x*t._x+this._y*t._y+this._z*t._z+this._w*t._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let t=this.length();return t===0?(this._x=0,this._y=0,this._z=0,this._w=1):(t=1/t,this._x=this._x*t,this._y=this._y*t,this._z=this._z*t,this._w=this._w*t),this._onChangeCallback(),this}multiply(t){return this.multiplyQuaternions(this,t)}premultiply(t){return this.multiplyQuaternions(t,this)}multiplyQuaternions(t,e){const n=t._x,s=t._y,r=t._z,a=t._w,o=e._x,l=e._y,c=e._z,h=e._w;return this._x=n*h+a*o+s*c-r*l,this._y=s*h+a*l+r*o-n*c,this._z=r*h+a*c+n*l-s*o,this._w=a*h-n*o-s*l-r*c,this._onChangeCallback(),this}slerp(t,e){if(e===0)return this;if(e===1)return this.copy(t);const n=this._x,s=this._y,r=this._z,a=this._w;let o=a*t._w+n*t._x+s*t._y+r*t._z;if(o<0?(this._w=-t._w,this._x=-t._x,this._y=-t._y,this._z=-t._z,o=-o):this.copy(t),o>=1)return this._w=a,this._x=n,this._y=s,this._z=r,this;const l=1-o*o;if(l<=Number.EPSILON){const u=1-e;return this._w=u*a+e*this._w,this._x=u*n+e*this._x,this._y=u*s+e*this._y,this._z=u*r+e*this._z,this.normalize(),this}const c=Math.sqrt(l),h=Math.atan2(c,o),d=Math.sin((1-e)*h)/c,p=Math.sin(e*h)/c;return this._w=a*d+this._w*p,this._x=n*d+this._x*p,this._y=s*d+this._y*p,this._z=r*d+this._z*p,this._onChangeCallback(),this}slerpQuaternions(t,e,n){return this.copy(t).slerp(e,n)}random(){const t=2*Math.PI*Math.random(),e=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(t),s*Math.cos(t),r*Math.sin(e),r*Math.cos(e))}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._w===this._w}fromArray(t,e=0){return this._x=t[e],this._y=t[e+1],this._z=t[e+2],this._w=t[e+3],this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._w,t}fromBufferAttribute(t,e){return this._x=t.getX(e),this._y=t.getY(e),this._z=t.getZ(e),this._w=t.getW(e),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class P{constructor(t=0,e=0,n=0){P.prototype.isVector3=!0,this.x=t,this.y=e,this.z=n}set(t,e,n){return n===void 0&&(n=this.z),this.x=t,this.y=e,this.z=n,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;default:throw new Error("index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this}multiplyVectors(t,e){return this.x=t.x*e.x,this.y=t.y*e.y,this.z=t.z*e.z,this}applyEuler(t){return this.applyQuaternion(ko.setFromEuler(t))}applyAxisAngle(t,e){return this.applyQuaternion(ko.setFromAxisAngle(t,e))}applyMatrix3(t){const e=this.x,n=this.y,s=this.z,r=t.elements;return this.x=r[0]*e+r[3]*n+r[6]*s,this.y=r[1]*e+r[4]*n+r[7]*s,this.z=r[2]*e+r[5]*n+r[8]*s,this}applyNormalMatrix(t){return this.applyMatrix3(t).normalize()}applyMatrix4(t){const e=this.x,n=this.y,s=this.z,r=t.elements,a=1/(r[3]*e+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*e+r[4]*n+r[8]*s+r[12])*a,this.y=(r[1]*e+r[5]*n+r[9]*s+r[13])*a,this.z=(r[2]*e+r[6]*n+r[10]*s+r[14])*a,this}applyQuaternion(t){const e=this.x,n=this.y,s=this.z,r=t.x,a=t.y,o=t.z,l=t.w,c=2*(a*s-o*n),h=2*(o*e-r*s),d=2*(r*n-a*e);return this.x=e+l*c+a*d-o*h,this.y=n+l*h+o*c-r*d,this.z=s+l*d+r*h-a*c,this}project(t){return this.applyMatrix4(t.matrixWorldInverse).applyMatrix4(t.projectionMatrix)}unproject(t){return this.applyMatrix4(t.projectionMatrixInverse).applyMatrix4(t.matrixWorld)}transformDirection(t){const e=this.x,n=this.y,s=this.z,r=t.elements;return this.x=r[0]*e+r[4]*n+r[8]*s,this.y=r[1]*e+r[5]*n+r[9]*s,this.z=r[2]*e+r[6]*n+r[10]*s,this.normalize()}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this}divideScalar(t){return this.multiplyScalar(1/t)}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this}clamp(t,e){return this.x=Yt(this.x,t.x,e.x),this.y=Yt(this.y,t.y,e.y),this.z=Yt(this.z,t.z,e.z),this}clampScalar(t,e){return this.x=Yt(this.x,t,e),this.y=Yt(this.y,t,e),this.z=Yt(this.z,t,e),this}clampLength(t,e){const n=this.length();return this.divideScalar(n||1).multiplyScalar(Yt(n,t,e))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this}cross(t){return this.crossVectors(this,t)}crossVectors(t,e){const n=t.x,s=t.y,r=t.z,a=e.x,o=e.y,l=e.z;return this.x=s*l-r*o,this.y=r*a-n*l,this.z=n*o-s*a,this}projectOnVector(t){const e=t.lengthSq();if(e===0)return this.set(0,0,0);const n=t.dot(this)/e;return this.copy(t).multiplyScalar(n)}projectOnPlane(t){return Vr.copy(this).projectOnVector(t),this.sub(Vr)}reflect(t){return this.sub(Vr.copy(t).multiplyScalar(2*this.dot(t)))}angleTo(t){const e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;const n=this.dot(t)/e;return Math.acos(Yt(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){const e=this.x-t.x,n=this.y-t.y,s=this.z-t.z;return e*e+n*n+s*s}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)+Math.abs(this.z-t.z)}setFromSpherical(t){return this.setFromSphericalCoords(t.radius,t.phi,t.theta)}setFromSphericalCoords(t,e,n){const s=Math.sin(e)*t;return this.x=s*Math.sin(n),this.y=Math.cos(e)*t,this.z=s*Math.cos(n),this}setFromCylindrical(t){return this.setFromCylindricalCoords(t.radius,t.theta,t.y)}setFromCylindricalCoords(t,e,n){return this.x=t*Math.sin(e),this.y=n,this.z=t*Math.cos(e),this}setFromMatrixPosition(t){const e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this}setFromMatrixScale(t){const e=this.setFromMatrixColumn(t,0).length(),n=this.setFromMatrixColumn(t,1).length(),s=this.setFromMatrixColumn(t,2).length();return this.x=e,this.y=n,this.z=s,this}setFromMatrixColumn(t,e){return this.fromArray(t.elements,e*4)}setFromMatrix3Column(t,e){return this.fromArray(t.elements,e*3)}setFromEuler(t){return this.x=t._x,this.y=t._y,this.z=t._z,this}setFromColor(t){return this.x=t.r,this.y=t.g,this.z=t.b,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const t=Math.random()*Math.PI*2,e=Math.random()*2-1,n=Math.sqrt(1-e*e);return this.x=n*Math.cos(t),this.y=e,this.z=n*Math.sin(t),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const Vr=new P,ko=new ci;class ui{constructor(t=new P(1/0,1/0,1/0),e=new P(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromArray(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e+=3)this.expandByPoint(cn.fromArray(t,e));return this}setFromBufferAttribute(t){this.makeEmpty();for(let e=0,n=t.count;e<n;e++)this.expandByPoint(cn.fromBufferAttribute(t,e));return this}setFromPoints(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){const n=cn.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(n),this.max.copy(t).add(n),this}setFromObject(t,e=!1){return this.makeEmpty(),this.expandByObject(t,e)}clone(){return new this.constructor().copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(t){return this.isEmpty()?t.set(0,0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}expandByObject(t,e=!1){t.updateWorldMatrix(!1,!1);const n=t.geometry;if(n!==void 0){const r=n.getAttribute("position");if(e===!0&&r!==void 0&&t.isInstancedMesh!==!0)for(let a=0,o=r.count;a<o;a++)t.isMesh===!0?t.getVertexPosition(a,cn):cn.fromBufferAttribute(r,a),cn.applyMatrix4(t.matrixWorld),this.expandByPoint(cn);else t.boundingBox!==void 0?(t.boundingBox===null&&t.computeBoundingBox(),Ns.copy(t.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Ns.copy(n.boundingBox)),Ns.applyMatrix4(t.matrixWorld),this.union(Ns)}const s=t.children;for(let r=0,a=s.length;r<a;r++)this.expandByObject(s[r],e);return this}containsPoint(t){return t.x>=this.min.x&&t.x<=this.max.x&&t.y>=this.min.y&&t.y<=this.max.y&&t.z>=this.min.z&&t.z<=this.max.z}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y&&this.min.z<=t.min.z&&t.max.z<=this.max.z}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y),(t.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(t){return t.max.x>=this.min.x&&t.min.x<=this.max.x&&t.max.y>=this.min.y&&t.min.y<=this.max.y&&t.max.z>=this.min.z&&t.min.z<=this.max.z}intersectsSphere(t){return this.clampPoint(t.center,cn),cn.distanceToSquared(t.center)<=t.radius*t.radius}intersectsPlane(t){let e,n;return t.normal.x>0?(e=t.normal.x*this.min.x,n=t.normal.x*this.max.x):(e=t.normal.x*this.max.x,n=t.normal.x*this.min.x),t.normal.y>0?(e+=t.normal.y*this.min.y,n+=t.normal.y*this.max.y):(e+=t.normal.y*this.max.y,n+=t.normal.y*this.min.y),t.normal.z>0?(e+=t.normal.z*this.min.z,n+=t.normal.z*this.max.z):(e+=t.normal.z*this.max.z,n+=t.normal.z*this.min.z),e<=-t.constant&&n>=-t.constant}intersectsTriangle(t){if(this.isEmpty())return!1;this.getCenter(ss),Fs.subVectors(this.max,ss),xi.subVectors(t.a,ss),Mi.subVectors(t.b,ss),Si.subVectors(t.c,ss),Un.subVectors(Mi,xi),In.subVectors(Si,Mi),Zn.subVectors(xi,Si);let e=[0,-Un.z,Un.y,0,-In.z,In.y,0,-Zn.z,Zn.y,Un.z,0,-Un.x,In.z,0,-In.x,Zn.z,0,-Zn.x,-Un.y,Un.x,0,-In.y,In.x,0,-Zn.y,Zn.x,0];return!Gr(e,xi,Mi,Si,Fs)||(e=[1,0,0,0,1,0,0,0,1],!Gr(e,xi,Mi,Si,Fs))?!1:(Os.crossVectors(Un,In),e=[Os.x,Os.y,Os.z],Gr(e,xi,Mi,Si,Fs))}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return this.clampPoint(t,cn).distanceTo(t)}getBoundingSphere(t){return this.isEmpty()?t.makeEmpty():(this.getCenter(t.center),t.radius=this.getSize(cn).length()*.5),t}intersect(t){return this.min.max(t.min),this.max.min(t.max),this.isEmpty()&&this.makeEmpty(),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}applyMatrix4(t){return this.isEmpty()?this:(Sn[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(t),Sn[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(t),Sn[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(t),Sn[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(t),Sn[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(t),Sn[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(t),Sn[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(t),Sn[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(t),this.setFromPoints(Sn),this)}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}}const Sn=[new P,new P,new P,new P,new P,new P,new P,new P],cn=new P,Ns=new ui,xi=new P,Mi=new P,Si=new P,Un=new P,In=new P,Zn=new P,ss=new P,Fs=new P,Os=new P,Kn=new P;function Gr(i,t,e,n,s){for(let r=0,a=i.length-3;r<=a;r+=3){Kn.fromArray(i,r);const o=s.x*Math.abs(Kn.x)+s.y*Math.abs(Kn.y)+s.z*Math.abs(Kn.z),l=t.dot(Kn),c=e.dot(Kn),h=n.dot(Kn);if(Math.max(-Math.max(l,c,h),Math.min(l,c,h))>o)return!1}return!0}const qh=new ui,rs=new P,Wr=new P;class di{constructor(t=new P,e=-1){this.isSphere=!0,this.center=t,this.radius=e}set(t,e){return this.center.copy(t),this.radius=e,this}setFromPoints(t,e){const n=this.center;e!==void 0?n.copy(e):qh.setFromPoints(t).getCenter(n);let s=0;for(let r=0,a=t.length;r<a;r++)s=Math.max(s,n.distanceToSquared(t[r]));return this.radius=Math.sqrt(s),this}copy(t){return this.center.copy(t.center),this.radius=t.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(t){return t.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(t){return t.distanceTo(this.center)-this.radius}intersectsSphere(t){const e=this.radius+t.radius;return t.center.distanceToSquared(this.center)<=e*e}intersectsBox(t){return t.intersectsSphere(this)}intersectsPlane(t){return Math.abs(t.distanceToPoint(this.center))<=this.radius}clampPoint(t,e){const n=this.center.distanceToSquared(t);return e.copy(t),n>this.radius*this.radius&&(e.sub(this.center).normalize(),e.multiplyScalar(this.radius).add(this.center)),e}getBoundingBox(t){return this.isEmpty()?(t.makeEmpty(),t):(t.set(this.center,this.center),t.expandByScalar(this.radius),t)}applyMatrix4(t){return this.center.applyMatrix4(t),this.radius=this.radius*t.getMaxScaleOnAxis(),this}translate(t){return this.center.add(t),this}expandByPoint(t){if(this.isEmpty())return this.center.copy(t),this.radius=0,this;rs.subVectors(t,this.center);const e=rs.lengthSq();if(e>this.radius*this.radius){const n=Math.sqrt(e),s=(n-this.radius)*.5;this.center.addScaledVector(rs,s/n),this.radius+=s}return this}union(t){return t.isEmpty()?this:this.isEmpty()?(this.copy(t),this):(this.center.equals(t.center)===!0?this.radius=Math.max(this.radius,t.radius):(Wr.subVectors(t.center,this.center).setLength(t.radius),this.expandByPoint(rs.copy(t.center).add(Wr)),this.expandByPoint(rs.copy(t.center).sub(Wr))),this)}equals(t){return t.center.equals(this.center)&&t.radius===this.radius}clone(){return new this.constructor().copy(this)}}const yn=new P,Xr=new P,Bs=new P,Nn=new P,Yr=new P,zs=new P,qr=new P;class bs{constructor(t=new P,e=new P(0,0,-1)){this.origin=t,this.direction=e}set(t,e){return this.origin.copy(t),this.direction.copy(e),this}copy(t){return this.origin.copy(t.origin),this.direction.copy(t.direction),this}at(t,e){return e.copy(this.origin).addScaledVector(this.direction,t)}lookAt(t){return this.direction.copy(t).sub(this.origin).normalize(),this}recast(t){return this.origin.copy(this.at(t,yn)),this}closestPointToPoint(t,e){e.subVectors(t,this.origin);const n=e.dot(this.direction);return n<0?e.copy(this.origin):e.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(t){return Math.sqrt(this.distanceSqToPoint(t))}distanceSqToPoint(t){const e=yn.subVectors(t,this.origin).dot(this.direction);return e<0?this.origin.distanceToSquared(t):(yn.copy(this.origin).addScaledVector(this.direction,e),yn.distanceToSquared(t))}distanceSqToSegment(t,e,n,s){Xr.copy(t).add(e).multiplyScalar(.5),Bs.copy(e).sub(t).normalize(),Nn.copy(this.origin).sub(Xr);const r=t.distanceTo(e)*.5,a=-this.direction.dot(Bs),o=Nn.dot(this.direction),l=-Nn.dot(Bs),c=Nn.lengthSq(),h=Math.abs(1-a*a);let d,p,u,g;if(h>0)if(d=a*l-o,p=a*o-l,g=r*h,d>=0)if(p>=-g)if(p<=g){const _=1/h;d*=_,p*=_,u=d*(d+a*p+2*o)+p*(a*d+p+2*l)+c}else p=r,d=Math.max(0,-(a*p+o)),u=-d*d+p*(p+2*l)+c;else p=-r,d=Math.max(0,-(a*p+o)),u=-d*d+p*(p+2*l)+c;else p<=-g?(d=Math.max(0,-(-a*r+o)),p=d>0?-r:Math.min(Math.max(-r,-l),r),u=-d*d+p*(p+2*l)+c):p<=g?(d=0,p=Math.min(Math.max(-r,-l),r),u=p*(p+2*l)+c):(d=Math.max(0,-(a*r+o)),p=d>0?r:Math.min(Math.max(-r,-l),r),u=-d*d+p*(p+2*l)+c);else p=a>0?-r:r,d=Math.max(0,-(a*p+o)),u=-d*d+p*(p+2*l)+c;return n&&n.copy(this.origin).addScaledVector(this.direction,d),s&&s.copy(Xr).addScaledVector(Bs,p),u}intersectSphere(t,e){yn.subVectors(t.center,this.origin);const n=yn.dot(this.direction),s=yn.dot(yn)-n*n,r=t.radius*t.radius;if(s>r)return null;const a=Math.sqrt(r-s),o=n-a,l=n+a;return l<0?null:o<0?this.at(l,e):this.at(o,e)}intersectsSphere(t){return this.distanceSqToPoint(t.center)<=t.radius*t.radius}distanceToPlane(t){const e=t.normal.dot(this.direction);if(e===0)return t.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(t.normal)+t.constant)/e;return n>=0?n:null}intersectPlane(t,e){const n=this.distanceToPlane(t);return n===null?null:this.at(n,e)}intersectsPlane(t){const e=t.distanceToPoint(this.origin);return e===0||t.normal.dot(this.direction)*e<0}intersectBox(t,e){let n,s,r,a,o,l;const c=1/this.direction.x,h=1/this.direction.y,d=1/this.direction.z,p=this.origin;return c>=0?(n=(t.min.x-p.x)*c,s=(t.max.x-p.x)*c):(n=(t.max.x-p.x)*c,s=(t.min.x-p.x)*c),h>=0?(r=(t.min.y-p.y)*h,a=(t.max.y-p.y)*h):(r=(t.max.y-p.y)*h,a=(t.min.y-p.y)*h),n>a||r>s||((r>n||isNaN(n))&&(n=r),(a<s||isNaN(s))&&(s=a),d>=0?(o=(t.min.z-p.z)*d,l=(t.max.z-p.z)*d):(o=(t.max.z-p.z)*d,l=(t.min.z-p.z)*d),n>l||o>s)||((o>n||n!==n)&&(n=o),(l<s||s!==s)&&(s=l),s<0)?null:this.at(n>=0?n:s,e)}intersectsBox(t){return this.intersectBox(t,yn)!==null}intersectTriangle(t,e,n,s,r){Yr.subVectors(e,t),zs.subVectors(n,t),qr.crossVectors(Yr,zs);let a=this.direction.dot(qr),o;if(a>0){if(s)return null;o=1}else if(a<0)o=-1,a=-a;else return null;Nn.subVectors(this.origin,t);const l=o*this.direction.dot(zs.crossVectors(Nn,zs));if(l<0)return null;const c=o*this.direction.dot(Yr.cross(Nn));if(c<0||l+c>a)return null;const h=-o*Nn.dot(qr);return h<0?null:this.at(h/a,r)}applyMatrix4(t){return this.origin.applyMatrix4(t),this.direction.transformDirection(t),this}equals(t){return t.origin.equals(this.origin)&&t.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class ne{constructor(t,e,n,s,r,a,o,l,c,h,d,p,u,g,_,m){ne.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],t!==void 0&&this.set(t,e,n,s,r,a,o,l,c,h,d,p,u,g,_,m)}set(t,e,n,s,r,a,o,l,c,h,d,p,u,g,_,m){const f=this.elements;return f[0]=t,f[4]=e,f[8]=n,f[12]=s,f[1]=r,f[5]=a,f[9]=o,f[13]=l,f[2]=c,f[6]=h,f[10]=d,f[14]=p,f[3]=u,f[7]=g,f[11]=_,f[15]=m,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new ne().fromArray(this.elements)}copy(t){const e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=n[11],e[12]=n[12],e[13]=n[13],e[14]=n[14],e[15]=n[15],this}copyPosition(t){const e=this.elements,n=t.elements;return e[12]=n[12],e[13]=n[13],e[14]=n[14],this}setFromMatrix3(t){const e=t.elements;return this.set(e[0],e[3],e[6],0,e[1],e[4],e[7],0,e[2],e[5],e[8],0,0,0,0,1),this}extractBasis(t,e,n){return t.setFromMatrixColumn(this,0),e.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this}makeBasis(t,e,n){return this.set(t.x,e.x,n.x,0,t.y,e.y,n.y,0,t.z,e.z,n.z,0,0,0,0,1),this}extractRotation(t){const e=this.elements,n=t.elements,s=1/yi.setFromMatrixColumn(t,0).length(),r=1/yi.setFromMatrixColumn(t,1).length(),a=1/yi.setFromMatrixColumn(t,2).length();return e[0]=n[0]*s,e[1]=n[1]*s,e[2]=n[2]*s,e[3]=0,e[4]=n[4]*r,e[5]=n[5]*r,e[6]=n[6]*r,e[7]=0,e[8]=n[8]*a,e[9]=n[9]*a,e[10]=n[10]*a,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromEuler(t){const e=this.elements,n=t.x,s=t.y,r=t.z,a=Math.cos(n),o=Math.sin(n),l=Math.cos(s),c=Math.sin(s),h=Math.cos(r),d=Math.sin(r);if(t.order==="XYZ"){const p=a*h,u=a*d,g=o*h,_=o*d;e[0]=l*h,e[4]=-l*d,e[8]=c,e[1]=u+g*c,e[5]=p-_*c,e[9]=-o*l,e[2]=_-p*c,e[6]=g+u*c,e[10]=a*l}else if(t.order==="YXZ"){const p=l*h,u=l*d,g=c*h,_=c*d;e[0]=p+_*o,e[4]=g*o-u,e[8]=a*c,e[1]=a*d,e[5]=a*h,e[9]=-o,e[2]=u*o-g,e[6]=_+p*o,e[10]=a*l}else if(t.order==="ZXY"){const p=l*h,u=l*d,g=c*h,_=c*d;e[0]=p-_*o,e[4]=-a*d,e[8]=g+u*o,e[1]=u+g*o,e[5]=a*h,e[9]=_-p*o,e[2]=-a*c,e[6]=o,e[10]=a*l}else if(t.order==="ZYX"){const p=a*h,u=a*d,g=o*h,_=o*d;e[0]=l*h,e[4]=g*c-u,e[8]=p*c+_,e[1]=l*d,e[5]=_*c+p,e[9]=u*c-g,e[2]=-c,e[6]=o*l,e[10]=a*l}else if(t.order==="YZX"){const p=a*l,u=a*c,g=o*l,_=o*c;e[0]=l*h,e[4]=_-p*d,e[8]=g*d+u,e[1]=d,e[5]=a*h,e[9]=-o*h,e[2]=-c*h,e[6]=u*d+g,e[10]=p-_*d}else if(t.order==="XZY"){const p=a*l,u=a*c,g=o*l,_=o*c;e[0]=l*h,e[4]=-d,e[8]=c*h,e[1]=p*d+_,e[5]=a*h,e[9]=u*d-g,e[2]=g*d-u,e[6]=o*h,e[10]=_*d+p}return e[3]=0,e[7]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromQuaternion(t){return this.compose(jh,t,Zh)}lookAt(t,e,n){const s=this.elements;return je.subVectors(t,e),je.lengthSq()===0&&(je.z=1),je.normalize(),Fn.crossVectors(n,je),Fn.lengthSq()===0&&(Math.abs(n.z)===1?je.x+=1e-4:je.z+=1e-4,je.normalize(),Fn.crossVectors(n,je)),Fn.normalize(),ks.crossVectors(je,Fn),s[0]=Fn.x,s[4]=ks.x,s[8]=je.x,s[1]=Fn.y,s[5]=ks.y,s[9]=je.y,s[2]=Fn.z,s[6]=ks.z,s[10]=je.z,this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){const n=t.elements,s=e.elements,r=this.elements,a=n[0],o=n[4],l=n[8],c=n[12],h=n[1],d=n[5],p=n[9],u=n[13],g=n[2],_=n[6],m=n[10],f=n[14],T=n[3],E=n[7],y=n[11],D=n[15],w=s[0],C=s[4],I=s[8],S=s[12],M=s[1],A=s[5],W=s[9],k=s[13],q=s[2],Q=s[6],X=s[10],tt=s[14],H=s[3],st=s[7],gt=s[11],Et=s[15];return r[0]=a*w+o*M+l*q+c*H,r[4]=a*C+o*A+l*Q+c*st,r[8]=a*I+o*W+l*X+c*gt,r[12]=a*S+o*k+l*tt+c*Et,r[1]=h*w+d*M+p*q+u*H,r[5]=h*C+d*A+p*Q+u*st,r[9]=h*I+d*W+p*X+u*gt,r[13]=h*S+d*k+p*tt+u*Et,r[2]=g*w+_*M+m*q+f*H,r[6]=g*C+_*A+m*Q+f*st,r[10]=g*I+_*W+m*X+f*gt,r[14]=g*S+_*k+m*tt+f*Et,r[3]=T*w+E*M+y*q+D*H,r[7]=T*C+E*A+y*Q+D*st,r[11]=T*I+E*W+y*X+D*gt,r[15]=T*S+E*k+y*tt+D*Et,this}multiplyScalar(t){const e=this.elements;return e[0]*=t,e[4]*=t,e[8]*=t,e[12]*=t,e[1]*=t,e[5]*=t,e[9]*=t,e[13]*=t,e[2]*=t,e[6]*=t,e[10]*=t,e[14]*=t,e[3]*=t,e[7]*=t,e[11]*=t,e[15]*=t,this}determinant(){const t=this.elements,e=t[0],n=t[4],s=t[8],r=t[12],a=t[1],o=t[5],l=t[9],c=t[13],h=t[2],d=t[6],p=t[10],u=t[14],g=t[3],_=t[7],m=t[11],f=t[15];return g*(+r*l*d-s*c*d-r*o*p+n*c*p+s*o*u-n*l*u)+_*(+e*l*u-e*c*p+r*a*p-s*a*u+s*c*h-r*l*h)+m*(+e*c*d-e*o*u-r*a*d+n*a*u+r*o*h-n*c*h)+f*(-s*o*h-e*l*d+e*o*p+s*a*d-n*a*p+n*l*h)}transpose(){const t=this.elements;let e;return e=t[1],t[1]=t[4],t[4]=e,e=t[2],t[2]=t[8],t[8]=e,e=t[6],t[6]=t[9],t[9]=e,e=t[3],t[3]=t[12],t[12]=e,e=t[7],t[7]=t[13],t[13]=e,e=t[11],t[11]=t[14],t[14]=e,this}setPosition(t,e,n){const s=this.elements;return t.isVector3?(s[12]=t.x,s[13]=t.y,s[14]=t.z):(s[12]=t,s[13]=e,s[14]=n),this}invert(){const t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],l=t[6],c=t[7],h=t[8],d=t[9],p=t[10],u=t[11],g=t[12],_=t[13],m=t[14],f=t[15],T=d*m*c-_*p*c+_*l*u-o*m*u-d*l*f+o*p*f,E=g*p*c-h*m*c-g*l*u+a*m*u+h*l*f-a*p*f,y=h*_*c-g*d*c+g*o*u-a*_*u-h*o*f+a*d*f,D=g*d*l-h*_*l-g*o*p+a*_*p+h*o*m-a*d*m,w=e*T+n*E+s*y+r*D;if(w===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const C=1/w;return t[0]=T*C,t[1]=(_*p*r-d*m*r-_*s*u+n*m*u+d*s*f-n*p*f)*C,t[2]=(o*m*r-_*l*r+_*s*c-n*m*c-o*s*f+n*l*f)*C,t[3]=(d*l*r-o*p*r-d*s*c+n*p*c+o*s*u-n*l*u)*C,t[4]=E*C,t[5]=(h*m*r-g*p*r+g*s*u-e*m*u-h*s*f+e*p*f)*C,t[6]=(g*l*r-a*m*r-g*s*c+e*m*c+a*s*f-e*l*f)*C,t[7]=(a*p*r-h*l*r+h*s*c-e*p*c-a*s*u+e*l*u)*C,t[8]=y*C,t[9]=(g*d*r-h*_*r-g*n*u+e*_*u+h*n*f-e*d*f)*C,t[10]=(a*_*r-g*o*r+g*n*c-e*_*c-a*n*f+e*o*f)*C,t[11]=(h*o*r-a*d*r-h*n*c+e*d*c+a*n*u-e*o*u)*C,t[12]=D*C,t[13]=(h*_*s-g*d*s+g*n*p-e*_*p-h*n*m+e*d*m)*C,t[14]=(g*o*s-a*_*s-g*n*l+e*_*l+a*n*m-e*o*m)*C,t[15]=(a*d*s-h*o*s+h*n*l-e*d*l-a*n*p+e*o*p)*C,this}scale(t){const e=this.elements,n=t.x,s=t.y,r=t.z;return e[0]*=n,e[4]*=s,e[8]*=r,e[1]*=n,e[5]*=s,e[9]*=r,e[2]*=n,e[6]*=s,e[10]*=r,e[3]*=n,e[7]*=s,e[11]*=r,this}getMaxScaleOnAxis(){const t=this.elements,e=t[0]*t[0]+t[1]*t[1]+t[2]*t[2],n=t[4]*t[4]+t[5]*t[5]+t[6]*t[6],s=t[8]*t[8]+t[9]*t[9]+t[10]*t[10];return Math.sqrt(Math.max(e,n,s))}makeTranslation(t,e,n){return t.isVector3?this.set(1,0,0,t.x,0,1,0,t.y,0,0,1,t.z,0,0,0,1):this.set(1,0,0,t,0,1,0,e,0,0,1,n,0,0,0,1),this}makeRotationX(t){const e=Math.cos(t),n=Math.sin(t);return this.set(1,0,0,0,0,e,-n,0,0,n,e,0,0,0,0,1),this}makeRotationY(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,0,n,0,0,1,0,0,-n,0,e,0,0,0,0,1),this}makeRotationZ(t){const e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,0,n,e,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(t,e){const n=Math.cos(e),s=Math.sin(e),r=1-n,a=t.x,o=t.y,l=t.z,c=r*a,h=r*o;return this.set(c*a+n,c*o-s*l,c*l+s*o,0,c*o+s*l,h*o+n,h*l-s*a,0,c*l-s*o,h*l+s*a,r*l*l+n,0,0,0,0,1),this}makeScale(t,e,n){return this.set(t,0,0,0,0,e,0,0,0,0,n,0,0,0,0,1),this}makeShear(t,e,n,s,r,a){return this.set(1,n,r,0,t,1,a,0,e,s,1,0,0,0,0,1),this}compose(t,e,n){const s=this.elements,r=e._x,a=e._y,o=e._z,l=e._w,c=r+r,h=a+a,d=o+o,p=r*c,u=r*h,g=r*d,_=a*h,m=a*d,f=o*d,T=l*c,E=l*h,y=l*d,D=n.x,w=n.y,C=n.z;return s[0]=(1-(_+f))*D,s[1]=(u+y)*D,s[2]=(g-E)*D,s[3]=0,s[4]=(u-y)*w,s[5]=(1-(p+f))*w,s[6]=(m+T)*w,s[7]=0,s[8]=(g+E)*C,s[9]=(m-T)*C,s[10]=(1-(p+_))*C,s[11]=0,s[12]=t.x,s[13]=t.y,s[14]=t.z,s[15]=1,this}decompose(t,e,n){const s=this.elements;let r=yi.set(s[0],s[1],s[2]).length();const a=yi.set(s[4],s[5],s[6]).length(),o=yi.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),t.x=s[12],t.y=s[13],t.z=s[14],hn.copy(this);const c=1/r,h=1/a,d=1/o;return hn.elements[0]*=c,hn.elements[1]*=c,hn.elements[2]*=c,hn.elements[4]*=h,hn.elements[5]*=h,hn.elements[6]*=h,hn.elements[8]*=d,hn.elements[9]*=d,hn.elements[10]*=d,e.setFromRotationMatrix(hn),n.x=r,n.y=a,n.z=o,this}makePerspective(t,e,n,s,r,a,o=Rn){const l=this.elements,c=2*r/(e-t),h=2*r/(n-s),d=(e+t)/(e-t),p=(n+s)/(n-s);let u,g;if(o===Rn)u=-(a+r)/(a-r),g=-2*a*r/(a-r);else if(o===Er)u=-a/(a-r),g=-a*r/(a-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return l[0]=c,l[4]=0,l[8]=d,l[12]=0,l[1]=0,l[5]=h,l[9]=p,l[13]=0,l[2]=0,l[6]=0,l[10]=u,l[14]=g,l[3]=0,l[7]=0,l[11]=-1,l[15]=0,this}makeOrthographic(t,e,n,s,r,a,o=Rn){const l=this.elements,c=1/(e-t),h=1/(n-s),d=1/(a-r),p=(e+t)*c,u=(n+s)*h;let g,_;if(o===Rn)g=(a+r)*d,_=-2*d;else if(o===Er)g=r*d,_=-1*d;else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return l[0]=2*c,l[4]=0,l[8]=0,l[12]=-p,l[1]=0,l[5]=2*h,l[9]=0,l[13]=-u,l[2]=0,l[6]=0,l[10]=_,l[14]=-g,l[3]=0,l[7]=0,l[11]=0,l[15]=1,this}equals(t){const e=this.elements,n=t.elements;for(let s=0;s<16;s++)if(e[s]!==n[s])return!1;return!0}fromArray(t,e=0){for(let n=0;n<16;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){const n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t[e+9]=n[9],t[e+10]=n[10],t[e+11]=n[11],t[e+12]=n[12],t[e+13]=n[13],t[e+14]=n[14],t[e+15]=n[15],t}}const yi=new P,hn=new ne,jh=new P(0,0,0),Zh=new P(1,1,1),Fn=new P,ks=new P,je=new P,Ho=new ne,Vo=new ci;class Mn{constructor(t=0,e=0,n=0,s=Mn.DEFAULT_ORDER){this.isEuler=!0,this._x=t,this._y=e,this._z=n,this._order=s}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get order(){return this._order}set order(t){this._order=t,this._onChangeCallback()}set(t,e,n,s=this._order){return this._x=t,this._y=e,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(t){return this._x=t._x,this._y=t._y,this._z=t._z,this._order=t._order,this._onChangeCallback(),this}setFromRotationMatrix(t,e=this._order,n=!0){const s=t.elements,r=s[0],a=s[4],o=s[8],l=s[1],c=s[5],h=s[9],d=s[2],p=s[6],u=s[10];switch(e){case"XYZ":this._y=Math.asin(Yt(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-h,u),this._z=Math.atan2(-a,r)):(this._x=Math.atan2(p,c),this._z=0);break;case"YXZ":this._x=Math.asin(-Yt(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(o,u),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-d,r),this._z=0);break;case"ZXY":this._x=Math.asin(Yt(p,-1,1)),Math.abs(p)<.9999999?(this._y=Math.atan2(-d,u),this._z=Math.atan2(-a,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-Yt(d,-1,1)),Math.abs(d)<.9999999?(this._x=Math.atan2(p,u),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-a,c));break;case"YZX":this._z=Math.asin(Yt(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-h,c),this._y=Math.atan2(-d,r)):(this._x=0,this._y=Math.atan2(o,u));break;case"XZY":this._z=Math.asin(-Yt(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(p,c),this._y=Math.atan2(o,r)):(this._x=Math.atan2(-h,u),this._y=0);break;default:console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "+e)}return this._order=e,n===!0&&this._onChangeCallback(),this}setFromQuaternion(t,e,n){return Ho.makeRotationFromQuaternion(t),this.setFromRotationMatrix(Ho,e,n)}setFromVector3(t,e=this._order){return this.set(t.x,t.y,t.z,e)}reorder(t){return Vo.setFromEuler(this),this.setFromQuaternion(Vo,t)}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._order===this._order}fromArray(t){return this._x=t[0],this._y=t[1],this._z=t[2],t[3]!==void 0&&(this._order=t[3]),this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._order,t}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Mn.DEFAULT_ORDER="XYZ";class wo{constructor(){this.mask=1}set(t){this.mask=(1<<t|0)>>>0}enable(t){this.mask|=1<<t|0}enableAll(){this.mask=-1}toggle(t){this.mask^=1<<t|0}disable(t){this.mask&=~(1<<t|0)}disableAll(){this.mask=0}test(t){return(this.mask&t.mask)!==0}isEnabled(t){return(this.mask&(1<<t|0))!==0}}let Kh=0;const Go=new P,Ei=new ci,En=new ne,Hs=new P,as=new P,$h=new P,Jh=new ci,Wo=new P(1,0,0),Xo=new P(0,1,0),Yo=new P(0,0,1),qo={type:"added"},Qh={type:"removed"},bi={type:"childadded",child:null},jr={type:"childremoved",child:null};class Ue extends hi{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:Kh++}),this.uuid=Xn(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Ue.DEFAULT_UP.clone();const t=new P,e=new Mn,n=new ci,s=new P(1,1,1);function r(){n.setFromEuler(e,!1)}function a(){e.setFromQuaternion(n,void 0,!1)}e._onChange(r),n._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:t},rotation:{configurable:!0,enumerable:!0,value:e},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new ne},normalMatrix:{value:new Ht}}),this.matrix=new ne,this.matrixWorld=new ne,this.matrixAutoUpdate=Ue.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Ue.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new wo,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(t){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(t),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(t){return this.quaternion.premultiply(t),this}setRotationFromAxisAngle(t,e){this.quaternion.setFromAxisAngle(t,e)}setRotationFromEuler(t){this.quaternion.setFromEuler(t,!0)}setRotationFromMatrix(t){this.quaternion.setFromRotationMatrix(t)}setRotationFromQuaternion(t){this.quaternion.copy(t)}rotateOnAxis(t,e){return Ei.setFromAxisAngle(t,e),this.quaternion.multiply(Ei),this}rotateOnWorldAxis(t,e){return Ei.setFromAxisAngle(t,e),this.quaternion.premultiply(Ei),this}rotateX(t){return this.rotateOnAxis(Wo,t)}rotateY(t){return this.rotateOnAxis(Xo,t)}rotateZ(t){return this.rotateOnAxis(Yo,t)}translateOnAxis(t,e){return Go.copy(t).applyQuaternion(this.quaternion),this.position.add(Go.multiplyScalar(e)),this}translateX(t){return this.translateOnAxis(Wo,t)}translateY(t){return this.translateOnAxis(Xo,t)}translateZ(t){return this.translateOnAxis(Yo,t)}localToWorld(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(this.matrixWorld)}worldToLocal(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(En.copy(this.matrixWorld).invert())}lookAt(t,e,n){t.isVector3?Hs.copy(t):Hs.set(t,e,n);const s=this.parent;this.updateWorldMatrix(!0,!1),as.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?En.lookAt(as,Hs,this.up):En.lookAt(Hs,as,this.up),this.quaternion.setFromRotationMatrix(En),s&&(En.extractRotation(s.matrixWorld),Ei.setFromRotationMatrix(En),this.quaternion.premultiply(Ei.invert()))}add(t){if(arguments.length>1){for(let e=0;e<arguments.length;e++)this.add(arguments[e]);return this}return t===this?(console.error("THREE.Object3D.add: object can't be added as a child of itself.",t),this):(t&&t.isObject3D?(t.removeFromParent(),t.parent=this,this.children.push(t),t.dispatchEvent(qo),bi.child=t,this.dispatchEvent(bi),bi.child=null):console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.",t),this)}remove(t){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const e=this.children.indexOf(t);return e!==-1&&(t.parent=null,this.children.splice(e,1),t.dispatchEvent(Qh),jr.child=t,this.dispatchEvent(jr),jr.child=null),this}removeFromParent(){const t=this.parent;return t!==null&&t.remove(this),this}clear(){return this.remove(...this.children)}attach(t){return this.updateWorldMatrix(!0,!1),En.copy(this.matrixWorld).invert(),t.parent!==null&&(t.parent.updateWorldMatrix(!0,!1),En.multiply(t.parent.matrixWorld)),t.applyMatrix4(En),t.removeFromParent(),t.parent=this,this.children.push(t),t.updateWorldMatrix(!1,!0),t.dispatchEvent(qo),bi.child=t,this.dispatchEvent(bi),bi.child=null,this}getObjectById(t){return this.getObjectByProperty("id",t)}getObjectByName(t){return this.getObjectByProperty("name",t)}getObjectByProperty(t,e){if(this[t]===e)return this;for(let n=0,s=this.children.length;n<s;n++){const a=this.children[n].getObjectByProperty(t,e);if(a!==void 0)return a}}getObjectsByProperty(t,e,n=[]){this[t]===e&&n.push(this);const s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].getObjectsByProperty(t,e,n);return n}getWorldPosition(t){return this.updateWorldMatrix(!0,!1),t.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(as,t,$h),t}getWorldScale(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(as,Jh,t),t}getWorldDirection(t){this.updateWorldMatrix(!0,!1);const e=this.matrixWorld.elements;return t.set(e[8],e[9],e[10]).normalize()}raycast(){}traverse(t){t(this);const e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].traverse(t)}traverseVisible(t){if(this.visible===!1)return;t(this);const e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].traverseVisible(t)}traverseAncestors(t){const e=this.parent;e!==null&&(t(e),e.traverseAncestors(t))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(t){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||t)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,t=!0);const e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].updateMatrixWorld(t)}updateWorldMatrix(t,e){const n=this.parent;if(t===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),e===!0){const s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(t){const e=t===void 0||typeof t=="string",n={};e&&(t={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.6,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.visibility=this._visibility,s.active=this._active,s.bounds=this._bounds.map(o=>({boxInitialized:o.boxInitialized,boxMin:o.box.min.toArray(),boxMax:o.box.max.toArray(),sphereInitialized:o.sphereInitialized,sphereRadius:o.sphere.radius,sphereCenter:o.sphere.center.toArray()})),s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.geometryCount=this._geometryCount,s.matricesTexture=this._matricesTexture.toJSON(t),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(t)),this.boundingSphere!==null&&(s.boundingSphere={center:s.boundingSphere.center.toArray(),radius:s.boundingSphere.radius}),this.boundingBox!==null&&(s.boundingBox={min:s.boundingBox.min.toArray(),max:s.boundingBox.max.toArray()}));function r(o,l){return o[l.uuid]===void 0&&(o[l.uuid]=l.toJSON(t)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(t).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(t).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(t.geometries,this.geometry);const o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){const l=o.shapes;if(Array.isArray(l))for(let c=0,h=l.length;c<h;c++){const d=l[c];r(t.shapes,d)}else r(t.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(t.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const o=[];for(let l=0,c=this.material.length;l<c;l++)o.push(r(t.materials,this.material[l]));s.material=o}else s.material=r(t.materials,this.material);if(this.children.length>0){s.children=[];for(let o=0;o<this.children.length;o++)s.children.push(this.children[o].toJSON(t).object)}if(this.animations.length>0){s.animations=[];for(let o=0;o<this.animations.length;o++){const l=this.animations[o];s.animations.push(r(t.animations,l))}}if(e){const o=a(t.geometries),l=a(t.materials),c=a(t.textures),h=a(t.images),d=a(t.shapes),p=a(t.skeletons),u=a(t.animations),g=a(t.nodes);o.length>0&&(n.geometries=o),l.length>0&&(n.materials=l),c.length>0&&(n.textures=c),h.length>0&&(n.images=h),d.length>0&&(n.shapes=d),p.length>0&&(n.skeletons=p),u.length>0&&(n.animations=u),g.length>0&&(n.nodes=g)}return n.object=s,n;function a(o){const l=[];for(const c in o){const h=o[c];delete h.metadata,l.push(h)}return l}}clone(t){return new this.constructor().copy(this,t)}copy(t,e=!0){if(this.name=t.name,this.up.copy(t.up),this.position.copy(t.position),this.rotation.order=t.rotation.order,this.quaternion.copy(t.quaternion),this.scale.copy(t.scale),this.matrix.copy(t.matrix),this.matrixWorld.copy(t.matrixWorld),this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrixWorldAutoUpdate=t.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=t.matrixWorldNeedsUpdate,this.layers.mask=t.layers.mask,this.visible=t.visible,this.castShadow=t.castShadow,this.receiveShadow=t.receiveShadow,this.frustumCulled=t.frustumCulled,this.renderOrder=t.renderOrder,this.animations=t.animations.slice(),this.userData=JSON.parse(JSON.stringify(t.userData)),e===!0)for(let n=0;n<t.children.length;n++){const s=t.children[n];this.add(s.clone())}return this}}Ue.DEFAULT_UP=new P(0,1,0);Ue.DEFAULT_MATRIX_AUTO_UPDATE=!0;Ue.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const un=new P,bn=new P,Zr=new P,Tn=new P,Ti=new P,wi=new P,jo=new P,Kr=new P,$r=new P,Jr=new P,Qr=new oe,ta=new oe,ea=new oe;class on{constructor(t=new P,e=new P,n=new P){this.a=t,this.b=e,this.c=n}static getNormal(t,e,n,s){s.subVectors(n,e),un.subVectors(t,e),s.cross(un);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(t,e,n,s,r){un.subVectors(s,e),bn.subVectors(n,e),Zr.subVectors(t,e);const a=un.dot(un),o=un.dot(bn),l=un.dot(Zr),c=bn.dot(bn),h=bn.dot(Zr),d=a*c-o*o;if(d===0)return r.set(0,0,0),null;const p=1/d,u=(c*l-o*h)*p,g=(a*h-o*l)*p;return r.set(1-u-g,g,u)}static containsPoint(t,e,n,s){return this.getBarycoord(t,e,n,s,Tn)===null?!1:Tn.x>=0&&Tn.y>=0&&Tn.x+Tn.y<=1}static getInterpolation(t,e,n,s,r,a,o,l){return this.getBarycoord(t,e,n,s,Tn)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,Tn.x),l.addScaledVector(a,Tn.y),l.addScaledVector(o,Tn.z),l)}static getInterpolatedAttribute(t,e,n,s,r,a){return Qr.setScalar(0),ta.setScalar(0),ea.setScalar(0),Qr.fromBufferAttribute(t,e),ta.fromBufferAttribute(t,n),ea.fromBufferAttribute(t,s),a.setScalar(0),a.addScaledVector(Qr,r.x),a.addScaledVector(ta,r.y),a.addScaledVector(ea,r.z),a}static isFrontFacing(t,e,n,s){return un.subVectors(n,e),bn.subVectors(t,e),un.cross(bn).dot(s)<0}set(t,e,n){return this.a.copy(t),this.b.copy(e),this.c.copy(n),this}setFromPointsAndIndices(t,e,n,s){return this.a.copy(t[e]),this.b.copy(t[n]),this.c.copy(t[s]),this}setFromAttributeAndIndices(t,e,n,s){return this.a.fromBufferAttribute(t,e),this.b.fromBufferAttribute(t,n),this.c.fromBufferAttribute(t,s),this}clone(){return new this.constructor().copy(this)}copy(t){return this.a.copy(t.a),this.b.copy(t.b),this.c.copy(t.c),this}getArea(){return un.subVectors(this.c,this.b),bn.subVectors(this.a,this.b),un.cross(bn).length()*.5}getMidpoint(t){return t.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(t){return on.getNormal(this.a,this.b,this.c,t)}getPlane(t){return t.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(t,e){return on.getBarycoord(t,this.a,this.b,this.c,e)}getInterpolation(t,e,n,s,r){return on.getInterpolation(t,this.a,this.b,this.c,e,n,s,r)}containsPoint(t){return on.containsPoint(t,this.a,this.b,this.c)}isFrontFacing(t){return on.isFrontFacing(this.a,this.b,this.c,t)}intersectsBox(t){return t.intersectsTriangle(this)}closestPointToPoint(t,e){const n=this.a,s=this.b,r=this.c;let a,o;Ti.subVectors(s,n),wi.subVectors(r,n),Kr.subVectors(t,n);const l=Ti.dot(Kr),c=wi.dot(Kr);if(l<=0&&c<=0)return e.copy(n);$r.subVectors(t,s);const h=Ti.dot($r),d=wi.dot($r);if(h>=0&&d<=h)return e.copy(s);const p=l*d-h*c;if(p<=0&&l>=0&&h<=0)return a=l/(l-h),e.copy(n).addScaledVector(Ti,a);Jr.subVectors(t,r);const u=Ti.dot(Jr),g=wi.dot(Jr);if(g>=0&&u<=g)return e.copy(r);const _=u*c-l*g;if(_<=0&&c>=0&&g<=0)return o=c/(c-g),e.copy(n).addScaledVector(wi,o);const m=h*g-u*d;if(m<=0&&d-h>=0&&u-g>=0)return jo.subVectors(r,s),o=(d-h)/(d-h+(u-g)),e.copy(s).addScaledVector(jo,o);const f=1/(m+_+p);return a=_*f,o=p*f,e.copy(n).addScaledVector(Ti,a).addScaledVector(wi,o)}equals(t){return t.a.equals(this.a)&&t.b.equals(this.b)&&t.c.equals(this.c)}}const _c={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},On={h:0,s:0,l:0},Vs={h:0,s:0,l:0};function na(i,t,e){return e<0&&(e+=1),e>1&&(e-=1),e<1/6?i+(t-i)*6*e:e<1/2?t:e<2/3?i+(t-i)*6*(2/3-e):i}class ot{constructor(t,e,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(t,e,n)}set(t,e,n){if(e===void 0&&n===void 0){const s=t;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(t,e,n);return this}setScalar(t){return this.r=t,this.g=t,this.b=t,this}setHex(t,e=an){return t=Math.floor(t),this.r=(t>>16&255)/255,this.g=(t>>8&255)/255,this.b=(t&255)/255,Qt.toWorkingColorSpace(this,e),this}setRGB(t,e,n,s=Qt.workingColorSpace){return this.r=t,this.g=e,this.b=n,Qt.toWorkingColorSpace(this,s),this}setHSL(t,e,n,s=Qt.workingColorSpace){if(t=Nh(t,1),e=Yt(e,0,1),n=Yt(n,0,1),e===0)this.r=this.g=this.b=n;else{const r=n<=.5?n*(1+e):n+e-n*e,a=2*n-r;this.r=na(a,r,t+1/3),this.g=na(a,r,t),this.b=na(a,r,t-1/3)}return Qt.toWorkingColorSpace(this,s),this}setStyle(t,e=an){function n(r){r!==void 0&&parseFloat(r)<1&&console.warn("THREE.Color: Alpha component of "+t+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(t)){let r;const a=s[1],o=s[2];switch(a){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,e);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,e);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,e);break;default:console.warn("THREE.Color: Unknown color model "+t)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(t)){const r=s[1],a=r.length;if(a===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,e);if(a===6)return this.setHex(parseInt(r,16),e);console.warn("THREE.Color: Invalid hex color "+t)}else if(t&&t.length>0)return this.setColorName(t,e);return this}setColorName(t,e=an){const n=_c[t.toLowerCase()];return n!==void 0?this.setHex(n,e):console.warn("THREE.Color: Unknown color "+t),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(t){return this.r=t.r,this.g=t.g,this.b=t.b,this}copySRGBToLinear(t){return this.r=Dn(t.r),this.g=Dn(t.g),this.b=Dn(t.b),this}copyLinearToSRGB(t){return this.r=Wi(t.r),this.g=Wi(t.g),this.b=Wi(t.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(t=an){return Qt.fromWorkingColorSpace(Ne.copy(this),t),Math.round(Yt(Ne.r*255,0,255))*65536+Math.round(Yt(Ne.g*255,0,255))*256+Math.round(Yt(Ne.b*255,0,255))}getHexString(t=an){return("000000"+this.getHex(t).toString(16)).slice(-6)}getHSL(t,e=Qt.workingColorSpace){Qt.fromWorkingColorSpace(Ne.copy(this),e);const n=Ne.r,s=Ne.g,r=Ne.b,a=Math.max(n,s,r),o=Math.min(n,s,r);let l,c;const h=(o+a)/2;if(o===a)l=0,c=0;else{const d=a-o;switch(c=h<=.5?d/(a+o):d/(2-a-o),a){case n:l=(s-r)/d+(s<r?6:0);break;case s:l=(r-n)/d+2;break;case r:l=(n-s)/d+4;break}l/=6}return t.h=l,t.s=c,t.l=h,t}getRGB(t,e=Qt.workingColorSpace){return Qt.fromWorkingColorSpace(Ne.copy(this),e),t.r=Ne.r,t.g=Ne.g,t.b=Ne.b,t}getStyle(t=an){Qt.fromWorkingColorSpace(Ne.copy(this),t);const e=Ne.r,n=Ne.g,s=Ne.b;return t!==an?`color(${t} ${e.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(e*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(t,e,n){return this.getHSL(On),this.setHSL(On.h+t,On.s+e,On.l+n)}add(t){return this.r+=t.r,this.g+=t.g,this.b+=t.b,this}addColors(t,e){return this.r=t.r+e.r,this.g=t.g+e.g,this.b=t.b+e.b,this}addScalar(t){return this.r+=t,this.g+=t,this.b+=t,this}sub(t){return this.r=Math.max(0,this.r-t.r),this.g=Math.max(0,this.g-t.g),this.b=Math.max(0,this.b-t.b),this}multiply(t){return this.r*=t.r,this.g*=t.g,this.b*=t.b,this}multiplyScalar(t){return this.r*=t,this.g*=t,this.b*=t,this}lerp(t,e){return this.r+=(t.r-this.r)*e,this.g+=(t.g-this.g)*e,this.b+=(t.b-this.b)*e,this}lerpColors(t,e,n){return this.r=t.r+(e.r-t.r)*n,this.g=t.g+(e.g-t.g)*n,this.b=t.b+(e.b-t.b)*n,this}lerpHSL(t,e){this.getHSL(On),t.getHSL(Vs);const n=zr(On.h,Vs.h,e),s=zr(On.s,Vs.s,e),r=zr(On.l,Vs.l,e);return this.setHSL(n,s,r),this}setFromVector3(t){return this.r=t.x,this.g=t.y,this.b=t.z,this}applyMatrix3(t){const e=this.r,n=this.g,s=this.b,r=t.elements;return this.r=r[0]*e+r[3]*n+r[6]*s,this.g=r[1]*e+r[4]*n+r[7]*s,this.b=r[2]*e+r[5]*n+r[8]*s,this}equals(t){return t.r===this.r&&t.g===this.g&&t.b===this.b}fromArray(t,e=0){return this.r=t[e],this.g=t[e+1],this.b=t[e+2],this}toArray(t=[],e=0){return t[e]=this.r,t[e+1]=this.g,t[e+2]=this.b,t}fromBufferAttribute(t,e){return this.r=t.getX(e),this.g=t.getY(e),this.b=t.getZ(e),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const Ne=new ot;ot.NAMES=_c;let tu=0;class qn extends hi{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:tu++}),this.uuid=Xn(),this.name="",this.type="Material",this.blending=Vi,this.side=Yn,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=ya,this.blendDst=Ea,this.blendEquation=ii,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new ot(0,0,0),this.blendAlpha=0,this.depthFunc=qi,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=No,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=_i,this.stencilZFail=_i,this.stencilZPass=_i,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(t){this._alphaTest>0!=t>0&&this.version++,this._alphaTest=t}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(t){if(t!==void 0)for(const e in t){const n=t[e];if(n===void 0){console.warn(`THREE.Material: parameter '${e}' has value of undefined.`);continue}const s=this[e];if(s===void 0){console.warn(`THREE.Material: '${e}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[e]=n}}toJSON(t){const e=t===void 0||typeof t=="string";e&&(t={textures:{},images:{}});const n={metadata:{version:4.6,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(t).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(t).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(t).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(t).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(t).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(t).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(t).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(t).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(t).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(t).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(t).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(t).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(t).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(t).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(t).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(t).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(t).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(t).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(t).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(t).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(t).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(t).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(t).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(t).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==Vi&&(n.blending=this.blending),this.side!==Yn&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==ya&&(n.blendSrc=this.blendSrc),this.blendDst!==Ea&&(n.blendDst=this.blendDst),this.blendEquation!==ii&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==qi&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==No&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==_i&&(n.stencilFail=this.stencilFail),this.stencilZFail!==_i&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==_i&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){const a=[];for(const o in r){const l=r[o];delete l.metadata,a.push(l)}return a}if(e){const r=s(t.textures),a=s(t.images);r.length>0&&(n.textures=r),a.length>0&&(n.images=a)}return n}clone(){return new this.constructor().copy(this)}copy(t){this.name=t.name,this.blending=t.blending,this.side=t.side,this.vertexColors=t.vertexColors,this.opacity=t.opacity,this.transparent=t.transparent,this.blendSrc=t.blendSrc,this.blendDst=t.blendDst,this.blendEquation=t.blendEquation,this.blendSrcAlpha=t.blendSrcAlpha,this.blendDstAlpha=t.blendDstAlpha,this.blendEquationAlpha=t.blendEquationAlpha,this.blendColor.copy(t.blendColor),this.blendAlpha=t.blendAlpha,this.depthFunc=t.depthFunc,this.depthTest=t.depthTest,this.depthWrite=t.depthWrite,this.stencilWriteMask=t.stencilWriteMask,this.stencilFunc=t.stencilFunc,this.stencilRef=t.stencilRef,this.stencilFuncMask=t.stencilFuncMask,this.stencilFail=t.stencilFail,this.stencilZFail=t.stencilZFail,this.stencilZPass=t.stencilZPass,this.stencilWrite=t.stencilWrite;const e=t.clippingPlanes;let n=null;if(e!==null){const s=e.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=e[r].clone()}return this.clippingPlanes=n,this.clipIntersection=t.clipIntersection,this.clipShadows=t.clipShadows,this.shadowSide=t.shadowSide,this.colorWrite=t.colorWrite,this.precision=t.precision,this.polygonOffset=t.polygonOffset,this.polygonOffsetFactor=t.polygonOffsetFactor,this.polygonOffsetUnits=t.polygonOffsetUnits,this.dithering=t.dithering,this.alphaTest=t.alphaTest,this.alphaHash=t.alphaHash,this.alphaToCoverage=t.alphaToCoverage,this.premultipliedAlpha=t.premultipliedAlpha,this.forceSinglePass=t.forceSinglePass,this.visible=t.visible,this.toneMapped=t.toneMapped,this.userData=JSON.parse(JSON.stringify(t.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(t){t===!0&&this.version++}onBuild(){console.warn("Material: onBuild() has been removed.")}}class Ss extends qn{constructor(t){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new ot(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Mn,this.combine=tc,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.fog=t.fog,this}}const Me=new P,Gs=new St;class he{constructor(t,e,n=!1){if(Array.isArray(t))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,this.name="",this.array=t,this.itemSize=e,this.count=t!==void 0?t.length/e:0,this.normalized=n,this.usage=oo,this.updateRanges=[],this.gpuType=xn,this.version=0}onUploadCallback(){}set needsUpdate(t){t===!0&&this.version++}setUsage(t){return this.usage=t,this}addUpdateRange(t,e){this.updateRanges.push({start:t,count:e})}clearUpdateRanges(){this.updateRanges.length=0}copy(t){return this.name=t.name,this.array=new t.array.constructor(t.array),this.itemSize=t.itemSize,this.count=t.count,this.normalized=t.normalized,this.usage=t.usage,this.gpuType=t.gpuType,this}copyAt(t,e,n){t*=this.itemSize,n*=e.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[t+s]=e.array[n+s];return this}copyArray(t){return this.array.set(t),this}applyMatrix3(t){if(this.itemSize===2)for(let e=0,n=this.count;e<n;e++)Gs.fromBufferAttribute(this,e),Gs.applyMatrix3(t),this.setXY(e,Gs.x,Gs.y);else if(this.itemSize===3)for(let e=0,n=this.count;e<n;e++)Me.fromBufferAttribute(this,e),Me.applyMatrix3(t),this.setXYZ(e,Me.x,Me.y,Me.z);return this}applyMatrix4(t){for(let e=0,n=this.count;e<n;e++)Me.fromBufferAttribute(this,e),Me.applyMatrix4(t),this.setXYZ(e,Me.x,Me.y,Me.z);return this}applyNormalMatrix(t){for(let e=0,n=this.count;e<n;e++)Me.fromBufferAttribute(this,e),Me.applyNormalMatrix(t),this.setXYZ(e,Me.x,Me.y,Me.z);return this}transformDirection(t){for(let e=0,n=this.count;e<n;e++)Me.fromBufferAttribute(this,e),Me.transformDirection(t),this.setXYZ(e,Me.x,Me.y,Me.z);return this}set(t,e=0){return this.array.set(t,e),this}getComponent(t,e){let n=this.array[t*this.itemSize+e];return this.normalized&&(n=_n(n,this.array)),n}setComponent(t,e,n){return this.normalized&&(n=re(n,this.array)),this.array[t*this.itemSize+e]=n,this}getX(t){let e=this.array[t*this.itemSize];return this.normalized&&(e=_n(e,this.array)),e}setX(t,e){return this.normalized&&(e=re(e,this.array)),this.array[t*this.itemSize]=e,this}getY(t){let e=this.array[t*this.itemSize+1];return this.normalized&&(e=_n(e,this.array)),e}setY(t,e){return this.normalized&&(e=re(e,this.array)),this.array[t*this.itemSize+1]=e,this}getZ(t){let e=this.array[t*this.itemSize+2];return this.normalized&&(e=_n(e,this.array)),e}setZ(t,e){return this.normalized&&(e=re(e,this.array)),this.array[t*this.itemSize+2]=e,this}getW(t){let e=this.array[t*this.itemSize+3];return this.normalized&&(e=_n(e,this.array)),e}setW(t,e){return this.normalized&&(e=re(e,this.array)),this.array[t*this.itemSize+3]=e,this}setXY(t,e,n){return t*=this.itemSize,this.normalized&&(e=re(e,this.array),n=re(n,this.array)),this.array[t+0]=e,this.array[t+1]=n,this}setXYZ(t,e,n,s){return t*=this.itemSize,this.normalized&&(e=re(e,this.array),n=re(n,this.array),s=re(s,this.array)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=s,this}setXYZW(t,e,n,s,r){return t*=this.itemSize,this.normalized&&(e=re(e,this.array),n=re(n,this.array),s=re(s,this.array),r=re(r,this.array)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=s,this.array[t+3]=r,this}onUpload(t){return this.onUploadCallback=t,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const t={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(t.name=this.name),this.usage!==oo&&(t.usage=this.usage),t}}class vc extends he{constructor(t,e,n){super(new Uint16Array(t),e,n)}}class xc extends he{constructor(t,e,n){super(new Uint32Array(t),e,n)}}class Oe extends he{constructor(t,e,n){super(new Float32Array(t),e,n)}}let eu=0;const rn=new ne,ia=new Ue,Ai=new P,Ze=new ui,os=new ui,Ce=new P;class ge extends hi{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:eu++}),this.uuid=Xn(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(t){return Array.isArray(t)?this.index=new(pc(t)?xc:vc)(t,1):this.index=t,this}setIndirect(t){return this.indirect=t,this}getIndirect(){return this.indirect}getAttribute(t){return this.attributes[t]}setAttribute(t,e){return this.attributes[t]=e,this}deleteAttribute(t){return delete this.attributes[t],this}hasAttribute(t){return this.attributes[t]!==void 0}addGroup(t,e,n=0){this.groups.push({start:t,count:e,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(t,e){this.drawRange.start=t,this.drawRange.count=e}applyMatrix4(t){const e=this.attributes.position;e!==void 0&&(e.applyMatrix4(t),e.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const r=new Ht().getNormalMatrix(t);n.applyNormalMatrix(r),n.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(t),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(t){return rn.makeRotationFromQuaternion(t),this.applyMatrix4(rn),this}rotateX(t){return rn.makeRotationX(t),this.applyMatrix4(rn),this}rotateY(t){return rn.makeRotationY(t),this.applyMatrix4(rn),this}rotateZ(t){return rn.makeRotationZ(t),this.applyMatrix4(rn),this}translate(t,e,n){return rn.makeTranslation(t,e,n),this.applyMatrix4(rn),this}scale(t,e,n){return rn.makeScale(t,e,n),this.applyMatrix4(rn),this}lookAt(t){return ia.lookAt(t),ia.updateMatrix(),this.applyMatrix4(ia.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Ai).negate(),this.translate(Ai.x,Ai.y,Ai.z),this}setFromPoints(t){const e=this.getAttribute("position");if(e===void 0){const n=[];for(let s=0,r=t.length;s<r;s++){const a=t[s];n.push(a.x,a.y,a.z||0)}this.setAttribute("position",new Oe(n,3))}else{const n=Math.min(t.length,e.count);for(let s=0;s<n;s++){const r=t[s];e.setXYZ(s,r.x,r.y,r.z||0)}t.length>e.count&&console.warn("THREE.BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),e.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new ui);const t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new P(-1/0,-1/0,-1/0),new P(1/0,1/0,1/0));return}if(t!==void 0){if(this.boundingBox.setFromBufferAttribute(t),e)for(let n=0,s=e.length;n<s;n++){const r=e[n];Ze.setFromBufferAttribute(r),this.morphTargetsRelative?(Ce.addVectors(this.boundingBox.min,Ze.min),this.boundingBox.expandByPoint(Ce),Ce.addVectors(this.boundingBox.max,Ze.max),this.boundingBox.expandByPoint(Ce)):(this.boundingBox.expandByPoint(Ze.min),this.boundingBox.expandByPoint(Ze.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new di);const t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new P,1/0);return}if(t){const n=this.boundingSphere.center;if(Ze.setFromBufferAttribute(t),e)for(let r=0,a=e.length;r<a;r++){const o=e[r];os.setFromBufferAttribute(o),this.morphTargetsRelative?(Ce.addVectors(Ze.min,os.min),Ze.expandByPoint(Ce),Ce.addVectors(Ze.max,os.max),Ze.expandByPoint(Ce)):(Ze.expandByPoint(os.min),Ze.expandByPoint(os.max))}Ze.getCenter(n);let s=0;for(let r=0,a=t.count;r<a;r++)Ce.fromBufferAttribute(t,r),s=Math.max(s,n.distanceToSquared(Ce));if(e)for(let r=0,a=e.length;r<a;r++){const o=e[r],l=this.morphTargetsRelative;for(let c=0,h=o.count;c<h;c++)Ce.fromBufferAttribute(o,c),l&&(Ai.fromBufferAttribute(t,c),Ce.add(Ai)),s=Math.max(s,n.distanceToSquared(Ce))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const t=this.index,e=this.attributes;if(t===null||e.position===void 0||e.normal===void 0||e.uv===void 0){console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=e.position,s=e.normal,r=e.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new he(new Float32Array(4*n.count),4));const a=this.getAttribute("tangent"),o=[],l=[];for(let I=0;I<n.count;I++)o[I]=new P,l[I]=new P;const c=new P,h=new P,d=new P,p=new St,u=new St,g=new St,_=new P,m=new P;function f(I,S,M){c.fromBufferAttribute(n,I),h.fromBufferAttribute(n,S),d.fromBufferAttribute(n,M),p.fromBufferAttribute(r,I),u.fromBufferAttribute(r,S),g.fromBufferAttribute(r,M),h.sub(c),d.sub(c),u.sub(p),g.sub(p);const A=1/(u.x*g.y-g.x*u.y);isFinite(A)&&(_.copy(h).multiplyScalar(g.y).addScaledVector(d,-u.y).multiplyScalar(A),m.copy(d).multiplyScalar(u.x).addScaledVector(h,-g.x).multiplyScalar(A),o[I].add(_),o[S].add(_),o[M].add(_),l[I].add(m),l[S].add(m),l[M].add(m))}let T=this.groups;T.length===0&&(T=[{start:0,count:t.count}]);for(let I=0,S=T.length;I<S;++I){const M=T[I],A=M.start,W=M.count;for(let k=A,q=A+W;k<q;k+=3)f(t.getX(k+0),t.getX(k+1),t.getX(k+2))}const E=new P,y=new P,D=new P,w=new P;function C(I){D.fromBufferAttribute(s,I),w.copy(D);const S=o[I];E.copy(S),E.sub(D.multiplyScalar(D.dot(S))).normalize(),y.crossVectors(w,S);const A=y.dot(l[I])<0?-1:1;a.setXYZW(I,E.x,E.y,E.z,A)}for(let I=0,S=T.length;I<S;++I){const M=T[I],A=M.start,W=M.count;for(let k=A,q=A+W;k<q;k+=3)C(t.getX(k+0)),C(t.getX(k+1)),C(t.getX(k+2))}}computeVertexNormals(){const t=this.index,e=this.getAttribute("position");if(e!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new he(new Float32Array(e.count*3),3),this.setAttribute("normal",n);else for(let p=0,u=n.count;p<u;p++)n.setXYZ(p,0,0,0);const s=new P,r=new P,a=new P,o=new P,l=new P,c=new P,h=new P,d=new P;if(t)for(let p=0,u=t.count;p<u;p+=3){const g=t.getX(p+0),_=t.getX(p+1),m=t.getX(p+2);s.fromBufferAttribute(e,g),r.fromBufferAttribute(e,_),a.fromBufferAttribute(e,m),h.subVectors(a,r),d.subVectors(s,r),h.cross(d),o.fromBufferAttribute(n,g),l.fromBufferAttribute(n,_),c.fromBufferAttribute(n,m),o.add(h),l.add(h),c.add(h),n.setXYZ(g,o.x,o.y,o.z),n.setXYZ(_,l.x,l.y,l.z),n.setXYZ(m,c.x,c.y,c.z)}else for(let p=0,u=e.count;p<u;p+=3)s.fromBufferAttribute(e,p+0),r.fromBufferAttribute(e,p+1),a.fromBufferAttribute(e,p+2),h.subVectors(a,r),d.subVectors(s,r),h.cross(d),n.setXYZ(p+0,h.x,h.y,h.z),n.setXYZ(p+1,h.x,h.y,h.z),n.setXYZ(p+2,h.x,h.y,h.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const t=this.attributes.normal;for(let e=0,n=t.count;e<n;e++)Ce.fromBufferAttribute(t,e),Ce.normalize(),t.setXYZ(e,Ce.x,Ce.y,Ce.z)}toNonIndexed(){function t(o,l){const c=o.array,h=o.itemSize,d=o.normalized,p=new c.constructor(l.length*h);let u=0,g=0;for(let _=0,m=l.length;_<m;_++){o.isInterleavedBufferAttribute?u=l[_]*o.data.stride+o.offset:u=l[_]*h;for(let f=0;f<h;f++)p[g++]=c[u++]}return new he(p,h,d)}if(this.index===null)return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const e=new ge,n=this.index.array,s=this.attributes;for(const o in s){const l=s[o],c=t(l,n);e.setAttribute(o,c)}const r=this.morphAttributes;for(const o in r){const l=[],c=r[o];for(let h=0,d=c.length;h<d;h++){const p=c[h],u=t(p,n);l.push(u)}e.morphAttributes[o]=l}e.morphTargetsRelative=this.morphTargetsRelative;const a=this.groups;for(let o=0,l=a.length;o<l;o++){const c=a[o];e.addGroup(c.start,c.count,c.materialIndex)}return e}toJSON(){const t={metadata:{version:4.6,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(t.uuid=this.uuid,t.type=this.type,this.name!==""&&(t.name=this.name),Object.keys(this.userData).length>0&&(t.userData=this.userData),this.parameters!==void 0){const l=this.parameters;for(const c in l)l[c]!==void 0&&(t[c]=l[c]);return t}t.data={attributes:{}};const e=this.index;e!==null&&(t.data.index={type:e.array.constructor.name,array:Array.prototype.slice.call(e.array)});const n=this.attributes;for(const l in n){const c=n[l];t.data.attributes[l]=c.toJSON(t.data)}const s={};let r=!1;for(const l in this.morphAttributes){const c=this.morphAttributes[l],h=[];for(let d=0,p=c.length;d<p;d++){const u=c[d];h.push(u.toJSON(t.data))}h.length>0&&(s[l]=h,r=!0)}r&&(t.data.morphAttributes=s,t.data.morphTargetsRelative=this.morphTargetsRelative);const a=this.groups;a.length>0&&(t.data.groups=JSON.parse(JSON.stringify(a)));const o=this.boundingSphere;return o!==null&&(t.data.boundingSphere={center:o.center.toArray(),radius:o.radius}),t}clone(){return new this.constructor().copy(this)}copy(t){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const e={};this.name=t.name;const n=t.index;n!==null&&this.setIndex(n.clone(e));const s=t.attributes;for(const c in s){const h=s[c];this.setAttribute(c,h.clone(e))}const r=t.morphAttributes;for(const c in r){const h=[],d=r[c];for(let p=0,u=d.length;p<u;p++)h.push(d[p].clone(e));this.morphAttributes[c]=h}this.morphTargetsRelative=t.morphTargetsRelative;const a=t.groups;for(let c=0,h=a.length;c<h;c++){const d=a[c];this.addGroup(d.start,d.count,d.materialIndex)}const o=t.boundingBox;o!==null&&(this.boundingBox=o.clone());const l=t.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=t.drawRange.start,this.drawRange.count=t.drawRange.count,this.userData=t.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const Zo=new ne,$n=new bs,Ws=new di,Ko=new P,Xs=new P,Ys=new P,qs=new P,sa=new P,js=new P,$o=new P,Zs=new P;class be extends Ue{constructor(t=new ge,e=new Ss){super(),this.isMesh=!0,this.type="Mesh",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),t.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=t.morphTargetInfluences.slice()),t.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},t.morphTargetDictionary)),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const s=e[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){const o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(t,e){const n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,a=n.morphTargetsRelative;e.fromBufferAttribute(s,t);const o=this.morphTargetInfluences;if(r&&o){js.set(0,0,0);for(let l=0,c=r.length;l<c;l++){const h=o[l],d=r[l];h!==0&&(sa.fromBufferAttribute(d,t),a?js.addScaledVector(sa,h):js.addScaledVector(sa.sub(e),h))}e.add(js)}return e}raycast(t,e){const n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Ws.copy(n.boundingSphere),Ws.applyMatrix4(r),$n.copy(t.ray).recast(t.near),!(Ws.containsPoint($n.origin)===!1&&($n.intersectSphere(Ws,Ko)===null||$n.origin.distanceToSquared(Ko)>(t.far-t.near)**2))&&(Zo.copy(r).invert(),$n.copy(t.ray).applyMatrix4(Zo),!(n.boundingBox!==null&&$n.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(t,e,$n)))}_computeIntersections(t,e,n){let s;const r=this.geometry,a=this.material,o=r.index,l=r.attributes.position,c=r.attributes.uv,h=r.attributes.uv1,d=r.attributes.normal,p=r.groups,u=r.drawRange;if(o!==null)if(Array.isArray(a))for(let g=0,_=p.length;g<_;g++){const m=p[g],f=a[m.materialIndex],T=Math.max(m.start,u.start),E=Math.min(o.count,Math.min(m.start+m.count,u.start+u.count));for(let y=T,D=E;y<D;y+=3){const w=o.getX(y),C=o.getX(y+1),I=o.getX(y+2);s=Ks(this,f,t,n,c,h,d,w,C,I),s&&(s.faceIndex=Math.floor(y/3),s.face.materialIndex=m.materialIndex,e.push(s))}}else{const g=Math.max(0,u.start),_=Math.min(o.count,u.start+u.count);for(let m=g,f=_;m<f;m+=3){const T=o.getX(m),E=o.getX(m+1),y=o.getX(m+2);s=Ks(this,a,t,n,c,h,d,T,E,y),s&&(s.faceIndex=Math.floor(m/3),e.push(s))}}else if(l!==void 0)if(Array.isArray(a))for(let g=0,_=p.length;g<_;g++){const m=p[g],f=a[m.materialIndex],T=Math.max(m.start,u.start),E=Math.min(l.count,Math.min(m.start+m.count,u.start+u.count));for(let y=T,D=E;y<D;y+=3){const w=y,C=y+1,I=y+2;s=Ks(this,f,t,n,c,h,d,w,C,I),s&&(s.faceIndex=Math.floor(y/3),s.face.materialIndex=m.materialIndex,e.push(s))}}else{const g=Math.max(0,u.start),_=Math.min(l.count,u.start+u.count);for(let m=g,f=_;m<f;m+=3){const T=m,E=m+1,y=m+2;s=Ks(this,a,t,n,c,h,d,T,E,y),s&&(s.faceIndex=Math.floor(m/3),e.push(s))}}}}function nu(i,t,e,n,s,r,a,o){let l;if(t.side===Xe?l=n.intersectTriangle(a,r,s,!0,o):l=n.intersectTriangle(s,r,a,t.side===Yn,o),l===null)return null;Zs.copy(o),Zs.applyMatrix4(i.matrixWorld);const c=e.ray.origin.distanceTo(Zs);return c<e.near||c>e.far?null:{distance:c,point:Zs.clone(),object:i}}function Ks(i,t,e,n,s,r,a,o,l,c){i.getVertexPosition(o,Xs),i.getVertexPosition(l,Ys),i.getVertexPosition(c,qs);const h=nu(i,t,e,n,Xs,Ys,qs,$o);if(h){const d=new P;on.getBarycoord($o,Xs,Ys,qs,d),s&&(h.uv=on.getInterpolatedAttribute(s,o,l,c,d,new St)),r&&(h.uv1=on.getInterpolatedAttribute(r,o,l,c,d,new St)),a&&(h.normal=on.getInterpolatedAttribute(a,o,l,c,d,new P),h.normal.dot(n.direction)>0&&h.normal.multiplyScalar(-1));const p={a:o,b:l,c,normal:new P,materialIndex:0};on.getNormal(Xs,Ys,qs,p.normal),h.face=p,h.barycoord=d}return h}class Ts extends ge{constructor(t=1,e=1,n=1,s=1,r=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:t,height:e,depth:n,widthSegments:s,heightSegments:r,depthSegments:a};const o=this;s=Math.floor(s),r=Math.floor(r),a=Math.floor(a);const l=[],c=[],h=[],d=[];let p=0,u=0;g("z","y","x",-1,-1,n,e,t,a,r,0),g("z","y","x",1,-1,n,e,-t,a,r,1),g("x","z","y",1,1,t,n,e,s,a,2),g("x","z","y",1,-1,t,n,-e,s,a,3),g("x","y","z",1,-1,t,e,n,s,r,4),g("x","y","z",-1,-1,t,e,-n,s,r,5),this.setIndex(l),this.setAttribute("position",new Oe(c,3)),this.setAttribute("normal",new Oe(h,3)),this.setAttribute("uv",new Oe(d,2));function g(_,m,f,T,E,y,D,w,C,I,S){const M=y/C,A=D/I,W=y/2,k=D/2,q=w/2,Q=C+1,X=I+1;let tt=0,H=0;const st=new P;for(let gt=0;gt<X;gt++){const Et=gt*A-k;for(let Ft=0;Ft<Q;Ft++){const Xt=Ft*M-W;st[_]=Xt*T,st[m]=Et*E,st[f]=q,c.push(st.x,st.y,st.z),st[_]=0,st[m]=0,st[f]=w>0?1:-1,h.push(st.x,st.y,st.z),d.push(Ft/C),d.push(1-gt/I),tt+=1}}for(let gt=0;gt<I;gt++)for(let Et=0;Et<C;Et++){const Ft=p+Et+Q*gt,Xt=p+Et+Q*(gt+1),Y=p+(Et+1)+Q*(gt+1),nt=p+(Et+1)+Q*gt;l.push(Ft,Xt,nt),l.push(Xt,Y,nt),H+=6}o.addGroup(u,H,S),u+=H,p+=tt}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new Ts(t.width,t.height,t.depth,t.widthSegments,t.heightSegments,t.depthSegments)}}function Qi(i){const t={};for(const e in i){t[e]={};for(const n in i[e]){const s=i[e][n];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),t[e][n]=null):t[e][n]=s.clone():Array.isArray(s)?t[e][n]=s.slice():t[e][n]=s}}return t}function ke(i){const t={};for(let e=0;e<i.length;e++){const n=Qi(i[e]);for(const s in n)t[s]=n[s]}return t}function iu(i){const t=[];for(let e=0;e<i.length;e++)t.push(i[e].clone());return t}function Mc(i){const t=i.getRenderTarget();return t===null?i.outputColorSpace:t.isXRRenderTarget===!0?t.texture.colorSpace:Qt.workingColorSpace}const Tr={clone:Qi,merge:ke};var su=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,ru=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class He extends qn{constructor(t){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=su,this.fragmentShader=ru,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,t!==void 0&&this.setValues(t)}copy(t){return super.copy(t),this.fragmentShader=t.fragmentShader,this.vertexShader=t.vertexShader,this.uniforms=Qi(t.uniforms),this.uniformsGroups=iu(t.uniformsGroups),this.defines=Object.assign({},t.defines),this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.fog=t.fog,this.lights=t.lights,this.clipping=t.clipping,this.extensions=Object.assign({},t.extensions),this.glslVersion=t.glslVersion,this}toJSON(t){const e=super.toJSON(t);e.glslVersion=this.glslVersion,e.uniforms={};for(const s in this.uniforms){const a=this.uniforms[s].value;a&&a.isTexture?e.uniforms[s]={type:"t",value:a.toJSON(t).uuid}:a&&a.isColor?e.uniforms[s]={type:"c",value:a.getHex()}:a&&a.isVector2?e.uniforms[s]={type:"v2",value:a.toArray()}:a&&a.isVector3?e.uniforms[s]={type:"v3",value:a.toArray()}:a&&a.isVector4?e.uniforms[s]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?e.uniforms[s]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?e.uniforms[s]={type:"m4",value:a.toArray()}:e.uniforms[s]={value:a}}Object.keys(this.defines).length>0&&(e.defines=this.defines),e.vertexShader=this.vertexShader,e.fragmentShader=this.fragmentShader,e.lights=this.lights,e.clipping=this.clipping;const n={};for(const s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(e.extensions=n),e}}class Sc extends Ue{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new ne,this.projectionMatrix=new ne,this.projectionMatrixInverse=new ne,this.coordinateSystem=Rn}copy(t,e){return super.copy(t,e),this.matrixWorldInverse.copy(t.matrixWorldInverse),this.projectionMatrix.copy(t.projectionMatrix),this.projectionMatrixInverse.copy(t.projectionMatrixInverse),this.coordinateSystem=t.coordinateSystem,this}getWorldDirection(t){return super.getWorldDirection(t).negate()}updateMatrixWorld(t){super.updateMatrixWorld(t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(t,e){super.updateWorldMatrix(t,e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const Bn=new P,Jo=new St,Qo=new St;class $e extends Sc{constructor(t=50,e=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=t,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=e,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.fov=t.fov,this.zoom=t.zoom,this.near=t.near,this.far=t.far,this.focus=t.focus,this.aspect=t.aspect,this.view=t.view===null?null:Object.assign({},t.view),this.filmGauge=t.filmGauge,this.filmOffset=t.filmOffset,this}setFocalLength(t){const e=.5*this.getFilmHeight()/t;this.fov=lo*2*Math.atan(e),this.updateProjectionMatrix()}getFocalLength(){const t=Math.tan(gr*.5*this.fov);return .5*this.getFilmHeight()/t}getEffectiveFOV(){return lo*2*Math.atan(Math.tan(gr*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(t,e,n){Bn.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),e.set(Bn.x,Bn.y).multiplyScalar(-t/Bn.z),Bn.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(Bn.x,Bn.y).multiplyScalar(-t/Bn.z)}getViewSize(t,e){return this.getViewBounds(t,Jo,Qo),e.subVectors(Qo,Jo)}setViewOffset(t,e,n,s,r,a){this.aspect=t/e,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=this.near;let e=t*Math.tan(gr*.5*this.fov)/this.zoom,n=2*e,s=this.aspect*n,r=-.5*s;const a=this.view;if(this.view!==null&&this.view.enabled){const l=a.fullWidth,c=a.fullHeight;r+=a.offsetX*s/l,e-=a.offsetY*n/c,s*=a.width/l,n*=a.height/c}const o=this.filmOffset;o!==0&&(r+=t*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,e,e-n,t,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.fov=this.fov,e.object.zoom=this.zoom,e.object.near=this.near,e.object.far=this.far,e.object.focus=this.focus,e.object.aspect=this.aspect,this.view!==null&&(e.object.view=Object.assign({},this.view)),e.object.filmGauge=this.filmGauge,e.object.filmOffset=this.filmOffset,e}}const Ri=-90,Ci=1;class au extends Ue{constructor(t,e,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new $e(Ri,Ci,t,e);s.layers=this.layers,this.add(s);const r=new $e(Ri,Ci,t,e);r.layers=this.layers,this.add(r);const a=new $e(Ri,Ci,t,e);a.layers=this.layers,this.add(a);const o=new $e(Ri,Ci,t,e);o.layers=this.layers,this.add(o);const l=new $e(Ri,Ci,t,e);l.layers=this.layers,this.add(l);const c=new $e(Ri,Ci,t,e);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){const t=this.coordinateSystem,e=this.children.concat(),[n,s,r,a,o,l]=e;for(const c of e)this.remove(c);if(t===Rn)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(t===Er)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+t);for(const c of e)this.add(c),c.updateMatrixWorld()}update(t,e){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==t.coordinateSystem&&(this.coordinateSystem=t.coordinateSystem,this.updateCoordinateSystem());const[r,a,o,l,c,h]=this.children,d=t.getRenderTarget(),p=t.getActiveCubeFace(),u=t.getActiveMipmapLevel(),g=t.xr.enabled;t.xr.enabled=!1;const _=n.texture.generateMipmaps;n.texture.generateMipmaps=!1,t.setRenderTarget(n,0,s),t.render(e,r),t.setRenderTarget(n,1,s),t.render(e,a),t.setRenderTarget(n,2,s),t.render(e,o),t.setRenderTarget(n,3,s),t.render(e,l),t.setRenderTarget(n,4,s),t.render(e,c),n.texture.generateMipmaps=_,t.setRenderTarget(n,5,s),t.render(e,h),t.setRenderTarget(d,p,u),t.xr.enabled=g,n.texture.needsPMREMUpdate=!0}}class yc extends Pe{constructor(t,e,n,s,r,a,o,l,c,h){t=t!==void 0?t:[],e=e!==void 0?e:ji,super(t,e,n,s,r,a,o,l,c,h),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(t){this.image=t}}class ou extends fn{constructor(t=1,e={}){super(t,t,e),this.isWebGLCubeRenderTarget=!0;const n={width:t,height:t,depth:1},s=[n,n,n,n,n,n];this.texture=new yc(s,e.mapping,e.wrapS,e.wrapT,e.magFilter,e.minFilter,e.format,e.type,e.anisotropy,e.colorSpace),this.texture.isRenderTargetTexture=!0,this.texture.generateMipmaps=e.generateMipmaps!==void 0?e.generateMipmaps:!1,this.texture.minFilter=e.minFilter!==void 0?e.minFilter:vn}fromEquirectangularTexture(t,e){this.texture.type=e.type,this.texture.colorSpace=e.colorSpace,this.texture.generateMipmaps=e.generateMipmaps,this.texture.minFilter=e.minFilter,this.texture.magFilter=e.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new Ts(5,5,5),r=new He({name:"CubemapFromEquirect",uniforms:Qi(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:Xe,blending:Cn});r.uniforms.tEquirect.value=e;const a=new be(s,r),o=e.minFilter;return e.minFilter===ai&&(e.minFilter=vn),new au(1,10,this).update(t,a),e.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(t,e,n,s){const r=t.getRenderTarget();for(let a=0;a<6;a++)t.setRenderTarget(this,a),t.clear(e,n,s);t.setRenderTarget(r)}}class Dr{constructor(t,e=25e-5){this.isFogExp2=!0,this.name="",this.color=new ot(t),this.density=e}clone(){return new Dr(this.color,this.density)}toJSON(){return{type:"FogExp2",name:this.name,color:this.color.getHex(),density:this.density}}}class lu extends Ue{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Mn,this.environmentIntensity=1,this.environmentRotation=new Mn,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(t,e){return super.copy(t,e),t.background!==null&&(this.background=t.background.clone()),t.environment!==null&&(this.environment=t.environment.clone()),t.fog!==null&&(this.fog=t.fog.clone()),this.backgroundBlurriness=t.backgroundBlurriness,this.backgroundIntensity=t.backgroundIntensity,this.backgroundRotation.copy(t.backgroundRotation),this.environmentIntensity=t.environmentIntensity,this.environmentRotation.copy(t.environmentRotation),t.overrideMaterial!==null&&(this.overrideMaterial=t.overrideMaterial.clone()),this.matrixAutoUpdate=t.matrixAutoUpdate,this}toJSON(t){const e=super.toJSON(t);return this.fog!==null&&(e.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(e.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(e.object.backgroundIntensity=this.backgroundIntensity),e.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(e.object.environmentIntensity=this.environmentIntensity),e.object.environmentRotation=this.environmentRotation.toArray(),e}}class cu{constructor(t,e){this.isInterleavedBuffer=!0,this.array=t,this.stride=e,this.count=t!==void 0?t.length/e:0,this.usage=oo,this.updateRanges=[],this.version=0,this.uuid=Xn()}onUploadCallback(){}set needsUpdate(t){t===!0&&this.version++}setUsage(t){return this.usage=t,this}addUpdateRange(t,e){this.updateRanges.push({start:t,count:e})}clearUpdateRanges(){this.updateRanges.length=0}copy(t){return this.array=new t.array.constructor(t.array),this.count=t.count,this.stride=t.stride,this.usage=t.usage,this}copyAt(t,e,n){t*=this.stride,n*=e.stride;for(let s=0,r=this.stride;s<r;s++)this.array[t+s]=e.array[n+s];return this}set(t,e=0){return this.array.set(t,e),this}clone(t){t.arrayBuffers===void 0&&(t.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=Xn()),t.arrayBuffers[this.array.buffer._uuid]===void 0&&(t.arrayBuffers[this.array.buffer._uuid]=this.array.slice(0).buffer);const e=new this.array.constructor(t.arrayBuffers[this.array.buffer._uuid]),n=new this.constructor(e,this.stride);return n.setUsage(this.usage),n}onUpload(t){return this.onUploadCallback=t,this}toJSON(t){return t.arrayBuffers===void 0&&(t.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=Xn()),t.arrayBuffers[this.array.buffer._uuid]===void 0&&(t.arrayBuffers[this.array.buffer._uuid]=Array.from(new Uint32Array(this.array.buffer))),{uuid:this.uuid,buffer:this.array.buffer._uuid,type:this.array.constructor.name,stride:this.stride}}}const ze=new P;class wr{constructor(t,e,n,s=!1){this.isInterleavedBufferAttribute=!0,this.name="",this.data=t,this.itemSize=e,this.offset=n,this.normalized=s}get count(){return this.data.count}get array(){return this.data.array}set needsUpdate(t){this.data.needsUpdate=t}applyMatrix4(t){for(let e=0,n=this.data.count;e<n;e++)ze.fromBufferAttribute(this,e),ze.applyMatrix4(t),this.setXYZ(e,ze.x,ze.y,ze.z);return this}applyNormalMatrix(t){for(let e=0,n=this.count;e<n;e++)ze.fromBufferAttribute(this,e),ze.applyNormalMatrix(t),this.setXYZ(e,ze.x,ze.y,ze.z);return this}transformDirection(t){for(let e=0,n=this.count;e<n;e++)ze.fromBufferAttribute(this,e),ze.transformDirection(t),this.setXYZ(e,ze.x,ze.y,ze.z);return this}getComponent(t,e){let n=this.array[t*this.data.stride+this.offset+e];return this.normalized&&(n=_n(n,this.array)),n}setComponent(t,e,n){return this.normalized&&(n=re(n,this.array)),this.data.array[t*this.data.stride+this.offset+e]=n,this}setX(t,e){return this.normalized&&(e=re(e,this.array)),this.data.array[t*this.data.stride+this.offset]=e,this}setY(t,e){return this.normalized&&(e=re(e,this.array)),this.data.array[t*this.data.stride+this.offset+1]=e,this}setZ(t,e){return this.normalized&&(e=re(e,this.array)),this.data.array[t*this.data.stride+this.offset+2]=e,this}setW(t,e){return this.normalized&&(e=re(e,this.array)),this.data.array[t*this.data.stride+this.offset+3]=e,this}getX(t){let e=this.data.array[t*this.data.stride+this.offset];return this.normalized&&(e=_n(e,this.array)),e}getY(t){let e=this.data.array[t*this.data.stride+this.offset+1];return this.normalized&&(e=_n(e,this.array)),e}getZ(t){let e=this.data.array[t*this.data.stride+this.offset+2];return this.normalized&&(e=_n(e,this.array)),e}getW(t){let e=this.data.array[t*this.data.stride+this.offset+3];return this.normalized&&(e=_n(e,this.array)),e}setXY(t,e,n){return t=t*this.data.stride+this.offset,this.normalized&&(e=re(e,this.array),n=re(n,this.array)),this.data.array[t+0]=e,this.data.array[t+1]=n,this}setXYZ(t,e,n,s){return t=t*this.data.stride+this.offset,this.normalized&&(e=re(e,this.array),n=re(n,this.array),s=re(s,this.array)),this.data.array[t+0]=e,this.data.array[t+1]=n,this.data.array[t+2]=s,this}setXYZW(t,e,n,s,r){return t=t*this.data.stride+this.offset,this.normalized&&(e=re(e,this.array),n=re(n,this.array),s=re(s,this.array),r=re(r,this.array)),this.data.array[t+0]=e,this.data.array[t+1]=n,this.data.array[t+2]=s,this.data.array[t+3]=r,this}clone(t){if(t===void 0){console.log("THREE.InterleavedBufferAttribute.clone(): Cloning an interleaved buffer attribute will de-interleave buffer data.");const e=[];for(let n=0;n<this.count;n++){const s=n*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)e.push(this.data.array[s+r])}return new he(new this.array.constructor(e),this.itemSize,this.normalized)}else return t.interleavedBuffers===void 0&&(t.interleavedBuffers={}),t.interleavedBuffers[this.data.uuid]===void 0&&(t.interleavedBuffers[this.data.uuid]=this.data.clone(t)),new wr(t.interleavedBuffers[this.data.uuid],this.itemSize,this.offset,this.normalized)}toJSON(t){if(t===void 0){console.log("THREE.InterleavedBufferAttribute.toJSON(): Serializing an interleaved buffer attribute will de-interleave buffer data.");const e=[];for(let n=0;n<this.count;n++){const s=n*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)e.push(this.data.array[s+r])}return{itemSize:this.itemSize,type:this.array.constructor.name,array:e,normalized:this.normalized}}else return t.interleavedBuffers===void 0&&(t.interleavedBuffers={}),t.interleavedBuffers[this.data.uuid]===void 0&&(t.interleavedBuffers[this.data.uuid]=this.data.toJSON(t)),{isInterleavedBufferAttribute:!0,itemSize:this.itemSize,data:this.data.uuid,offset:this.offset,normalized:this.normalized}}}class Xi extends qn{constructor(t){super(),this.isSpriteMaterial=!0,this.type="SpriteMaterial",this.color=new ot(16777215),this.map=null,this.alphaMap=null,this.rotation=0,this.sizeAttenuation=!0,this.transparent=!0,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.alphaMap=t.alphaMap,this.rotation=t.rotation,this.sizeAttenuation=t.sizeAttenuation,this.fog=t.fog,this}}let Pi;const ls=new P,Di=new P,Li=new P,Ui=new St,cs=new St,Ec=new ne,$s=new P,hs=new P,Js=new P,tl=new St,ra=new St,el=new St;class Bi extends Ue{constructor(t=new Xi){if(super(),this.isSprite=!0,this.type="Sprite",Pi===void 0){Pi=new ge;const e=new Float32Array([-.5,-.5,0,0,0,.5,-.5,0,1,0,.5,.5,0,1,1,-.5,.5,0,0,1]),n=new cu(e,5);Pi.setIndex([0,1,2,0,2,3]),Pi.setAttribute("position",new wr(n,3,0,!1)),Pi.setAttribute("uv",new wr(n,2,3,!1))}this.geometry=Pi,this.material=t,this.center=new St(.5,.5)}raycast(t,e){t.camera===null&&console.error('THREE.Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.'),Di.setFromMatrixScale(this.matrixWorld),Ec.copy(t.camera.matrixWorld),this.modelViewMatrix.multiplyMatrices(t.camera.matrixWorldInverse,this.matrixWorld),Li.setFromMatrixPosition(this.modelViewMatrix),t.camera.isPerspectiveCamera&&this.material.sizeAttenuation===!1&&Di.multiplyScalar(-Li.z);const n=this.material.rotation;let s,r;n!==0&&(r=Math.cos(n),s=Math.sin(n));const a=this.center;Qs($s.set(-.5,-.5,0),Li,a,Di,s,r),Qs(hs.set(.5,-.5,0),Li,a,Di,s,r),Qs(Js.set(.5,.5,0),Li,a,Di,s,r),tl.set(0,0),ra.set(1,0),el.set(1,1);let o=t.ray.intersectTriangle($s,hs,Js,!1,ls);if(o===null&&(Qs(hs.set(-.5,.5,0),Li,a,Di,s,r),ra.set(0,1),o=t.ray.intersectTriangle($s,Js,hs,!1,ls),o===null))return;const l=t.ray.origin.distanceTo(ls);l<t.near||l>t.far||e.push({distance:l,point:ls.clone(),uv:on.getInterpolation(ls,$s,hs,Js,tl,ra,el,new St),face:null,object:this})}copy(t,e){return super.copy(t,e),t.center!==void 0&&this.center.copy(t.center),this.material=t.material,this}}function Qs(i,t,e,n,s,r){Ui.subVectors(i,e).addScalar(.5).multiply(n),s!==void 0?(cs.x=r*Ui.x-s*Ui.y,cs.y=s*Ui.x+r*Ui.y):cs.copy(Ui),i.copy(t),i.x+=cs.x,i.y+=cs.y,i.applyMatrix4(Ec)}class hu extends Pe{constructor(t=null,e=1,n=1,s,r,a,o,l,c=Je,h=Je,d,p){super(null,a,o,l,c,h,s,r,d,p),this.isDataTexture=!0,this.image={data:t,width:e,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class nl extends he{constructor(t,e,n,s=1){super(t,e,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(t){return super.copy(t),this.meshPerAttribute=t.meshPerAttribute,this}toJSON(){const t=super.toJSON();return t.meshPerAttribute=this.meshPerAttribute,t.isInstancedBufferAttribute=!0,t}}const Ii=new ne,il=new ne,tr=[],sl=new ui,uu=new ne,us=new be,ds=new di;class du extends be{constructor(t,e,n){super(t,e),this.isInstancedMesh=!0,this.instanceMatrix=new nl(new Float32Array(n*16),16),this.instanceColor=null,this.morphTexture=null,this.count=n,this.boundingBox=null,this.boundingSphere=null;for(let s=0;s<n;s++)this.setMatrixAt(s,uu)}computeBoundingBox(){const t=this.geometry,e=this.count;this.boundingBox===null&&(this.boundingBox=new ui),t.boundingBox===null&&t.computeBoundingBox(),this.boundingBox.makeEmpty();for(let n=0;n<e;n++)this.getMatrixAt(n,Ii),sl.copy(t.boundingBox).applyMatrix4(Ii),this.boundingBox.union(sl)}computeBoundingSphere(){const t=this.geometry,e=this.count;this.boundingSphere===null&&(this.boundingSphere=new di),t.boundingSphere===null&&t.computeBoundingSphere(),this.boundingSphere.makeEmpty();for(let n=0;n<e;n++)this.getMatrixAt(n,Ii),ds.copy(t.boundingSphere).applyMatrix4(Ii),this.boundingSphere.union(ds)}copy(t,e){return super.copy(t,e),this.instanceMatrix.copy(t.instanceMatrix),t.morphTexture!==null&&(this.morphTexture=t.morphTexture.clone()),t.instanceColor!==null&&(this.instanceColor=t.instanceColor.clone()),this.count=t.count,t.boundingBox!==null&&(this.boundingBox=t.boundingBox.clone()),t.boundingSphere!==null&&(this.boundingSphere=t.boundingSphere.clone()),this}getColorAt(t,e){e.fromArray(this.instanceColor.array,t*3)}getMatrixAt(t,e){e.fromArray(this.instanceMatrix.array,t*16)}getMorphAt(t,e){const n=e.morphTargetInfluences,s=this.morphTexture.source.data.data,r=n.length+1,a=t*r+1;for(let o=0;o<n.length;o++)n[o]=s[a+o]}raycast(t,e){const n=this.matrixWorld,s=this.count;if(us.geometry=this.geometry,us.material=this.material,us.material!==void 0&&(this.boundingSphere===null&&this.computeBoundingSphere(),ds.copy(this.boundingSphere),ds.applyMatrix4(n),t.ray.intersectsSphere(ds)!==!1))for(let r=0;r<s;r++){this.getMatrixAt(r,Ii),il.multiplyMatrices(n,Ii),us.matrixWorld=il,us.raycast(t,tr);for(let a=0,o=tr.length;a<o;a++){const l=tr[a];l.instanceId=r,l.object=this,e.push(l)}tr.length=0}}setColorAt(t,e){this.instanceColor===null&&(this.instanceColor=new nl(new Float32Array(this.instanceMatrix.count*3).fill(1),3)),e.toArray(this.instanceColor.array,t*3)}setMatrixAt(t,e){e.toArray(this.instanceMatrix.array,t*16)}setMorphAt(t,e){const n=e.morphTargetInfluences,s=n.length+1;this.morphTexture===null&&(this.morphTexture=new hu(new Float32Array(s*this.count),s,this.count,yo,xn));const r=this.morphTexture.source.data.data;let a=0;for(let c=0;c<n.length;c++)a+=n[c];const o=this.geometry.morphTargetsRelative?1:1-a,l=s*t;r[l]=o,r.set(n,l+1)}updateMorphTargets(){}dispose(){return this.dispatchEvent({type:"dispose"}),this.morphTexture!==null&&(this.morphTexture.dispose(),this.morphTexture=null),this}}const aa=new P,fu=new P,pu=new Ht;class Hn{constructor(t=new P(1,0,0),e=0){this.isPlane=!0,this.normal=t,this.constant=e}set(t,e){return this.normal.copy(t),this.constant=e,this}setComponents(t,e,n,s){return this.normal.set(t,e,n),this.constant=s,this}setFromNormalAndCoplanarPoint(t,e){return this.normal.copy(t),this.constant=-e.dot(this.normal),this}setFromCoplanarPoints(t,e,n){const s=aa.subVectors(n,e).cross(fu.subVectors(t,e)).normalize();return this.setFromNormalAndCoplanarPoint(s,t),this}copy(t){return this.normal.copy(t.normal),this.constant=t.constant,this}normalize(){const t=1/this.normal.length();return this.normal.multiplyScalar(t),this.constant*=t,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(t){return this.normal.dot(t)+this.constant}distanceToSphere(t){return this.distanceToPoint(t.center)-t.radius}projectPoint(t,e){return e.copy(t).addScaledVector(this.normal,-this.distanceToPoint(t))}intersectLine(t,e){const n=t.delta(aa),s=this.normal.dot(n);if(s===0)return this.distanceToPoint(t.start)===0?e.copy(t.start):null;const r=-(t.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:e.copy(t.start).addScaledVector(n,r)}intersectsLine(t){const e=this.distanceToPoint(t.start),n=this.distanceToPoint(t.end);return e<0&&n>0||n<0&&e>0}intersectsBox(t){return t.intersectsPlane(this)}intersectsSphere(t){return t.intersectsPlane(this)}coplanarPoint(t){return t.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(t,e){const n=e||pu.getNormalMatrix(t),s=this.coplanarPoint(aa).applyMatrix4(t),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(t){return this.constant-=t.dot(this.normal),this}equals(t){return t.normal.equals(this.normal)&&t.constant===this.constant}clone(){return new this.constructor().copy(this)}}const Jn=new di,er=new P;class Ao{constructor(t=new Hn,e=new Hn,n=new Hn,s=new Hn,r=new Hn,a=new Hn){this.planes=[t,e,n,s,r,a]}set(t,e,n,s,r,a){const o=this.planes;return o[0].copy(t),o[1].copy(e),o[2].copy(n),o[3].copy(s),o[4].copy(r),o[5].copy(a),this}copy(t){const e=this.planes;for(let n=0;n<6;n++)e[n].copy(t.planes[n]);return this}setFromProjectionMatrix(t,e=Rn){const n=this.planes,s=t.elements,r=s[0],a=s[1],o=s[2],l=s[3],c=s[4],h=s[5],d=s[6],p=s[7],u=s[8],g=s[9],_=s[10],m=s[11],f=s[12],T=s[13],E=s[14],y=s[15];if(n[0].setComponents(l-r,p-c,m-u,y-f).normalize(),n[1].setComponents(l+r,p+c,m+u,y+f).normalize(),n[2].setComponents(l+a,p+h,m+g,y+T).normalize(),n[3].setComponents(l-a,p-h,m-g,y-T).normalize(),n[4].setComponents(l-o,p-d,m-_,y-E).normalize(),e===Rn)n[5].setComponents(l+o,p+d,m+_,y+E).normalize();else if(e===Er)n[5].setComponents(o,d,_,E).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+e);return this}intersectsObject(t){if(t.boundingSphere!==void 0)t.boundingSphere===null&&t.computeBoundingSphere(),Jn.copy(t.boundingSphere).applyMatrix4(t.matrixWorld);else{const e=t.geometry;e.boundingSphere===null&&e.computeBoundingSphere(),Jn.copy(e.boundingSphere).applyMatrix4(t.matrixWorld)}return this.intersectsSphere(Jn)}intersectsSprite(t){return Jn.center.set(0,0,0),Jn.radius=.7071067811865476,Jn.applyMatrix4(t.matrixWorld),this.intersectsSphere(Jn)}intersectsSphere(t){const e=this.planes,n=t.center,s=-t.radius;for(let r=0;r<6;r++)if(e[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(t){const e=this.planes;for(let n=0;n<6;n++){const s=e[n];if(er.x=s.normal.x>0?t.max.x:t.min.x,er.y=s.normal.y>0?t.max.y:t.min.y,er.z=s.normal.z>0?t.max.z:t.min.z,s.distanceToPoint(er)<0)return!1}return!0}containsPoint(t){const e=this.planes;for(let n=0;n<6;n++)if(e[n].distanceToPoint(t)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class Ar extends qn{constructor(t){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new ot(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.linewidth=t.linewidth,this.linecap=t.linecap,this.linejoin=t.linejoin,this.fog=t.fog,this}}const Rr=new P,Cr=new P,rl=new ne,fs=new bs,nr=new di,oa=new P,al=new P;class co extends Ue{constructor(t=new ge,e=new Ar){super(),this.isLine=!0,this.type="Line",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}computeLineDistances(){const t=this.geometry;if(t.index===null){const e=t.attributes.position,n=[0];for(let s=1,r=e.count;s<r;s++)Rr.fromBufferAttribute(e,s-1),Cr.fromBufferAttribute(e,s),n[s]=n[s-1],n[s]+=Rr.distanceTo(Cr);t.setAttribute("lineDistance",new Oe(n,1))}else console.warn("THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(t,e){const n=this.geometry,s=this.matrixWorld,r=t.params.Line.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),nr.copy(n.boundingSphere),nr.applyMatrix4(s),nr.radius+=r,t.ray.intersectsSphere(nr)===!1)return;rl.copy(s).invert(),fs.copy(t.ray).applyMatrix4(rl);const o=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=o*o,c=this.isLineSegments?2:1,h=n.index,p=n.attributes.position;if(h!==null){const u=Math.max(0,a.start),g=Math.min(h.count,a.start+a.count);for(let _=u,m=g-1;_<m;_+=c){const f=h.getX(_),T=h.getX(_+1),E=ir(this,t,fs,l,f,T);E&&e.push(E)}if(this.isLineLoop){const _=h.getX(g-1),m=h.getX(u),f=ir(this,t,fs,l,_,m);f&&e.push(f)}}else{const u=Math.max(0,a.start),g=Math.min(p.count,a.start+a.count);for(let _=u,m=g-1;_<m;_+=c){const f=ir(this,t,fs,l,_,_+1);f&&e.push(f)}if(this.isLineLoop){const _=ir(this,t,fs,l,g-1,u);_&&e.push(_)}}}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const s=e[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){const o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}}function ir(i,t,e,n,s,r){const a=i.geometry.attributes.position;if(Rr.fromBufferAttribute(a,s),Cr.fromBufferAttribute(a,r),e.distanceSqToSegment(Rr,Cr,oa,al)>n)return;oa.applyMatrix4(i.matrixWorld);const l=t.ray.origin.distanceTo(oa);if(!(l<t.near||l>t.far))return{distance:l,point:al.clone().applyMatrix4(i.matrixWorld),index:s,face:null,faceIndex:null,barycoord:null,object:i}}class oi extends qn{constructor(t){super(),this.isPointsMaterial=!0,this.type="PointsMaterial",this.color=new ot(16777215),this.map=null,this.alphaMap=null,this.size=1,this.sizeAttenuation=!0,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.alphaMap=t.alphaMap,this.size=t.size,this.sizeAttenuation=t.sizeAttenuation,this.fog=t.fog,this}}const ol=new ne,ho=new bs,sr=new di,rr=new P;class Yi extends Ue{constructor(t=new ge,e=new oi){super(),this.isPoints=!0,this.type="Points",this.geometry=t,this.material=e,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}raycast(t,e){const n=this.geometry,s=this.matrixWorld,r=t.params.Points.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),sr.copy(n.boundingSphere),sr.applyMatrix4(s),sr.radius+=r,t.ray.intersectsSphere(sr)===!1)return;ol.copy(s).invert(),ho.copy(t.ray).applyMatrix4(ol);const o=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=o*o,c=n.index,d=n.attributes.position;if(c!==null){const p=Math.max(0,a.start),u=Math.min(c.count,a.start+a.count);for(let g=p,_=u;g<_;g++){const m=c.getX(g);rr.fromBufferAttribute(d,m),ll(rr,m,l,s,t,e,this)}}else{const p=Math.max(0,a.start),u=Math.min(d.count,a.start+a.count);for(let g=p,_=u;g<_;g++)rr.fromBufferAttribute(d,g),ll(rr,g,l,s,t,e,this)}}updateMorphTargets(){const e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){const s=e[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){const o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}}function ll(i,t,e,n,s,r,a){const o=ho.distanceSqToPoint(i);if(o<e){const l=new P;ho.closestPointToPoint(i,l),l.applyMatrix4(n);const c=s.ray.origin.distanceTo(l);if(c<s.near||c>s.far)return;r.push({distance:c,distanceToRay:Math.sqrt(o),point:l,index:t,face:null,faceIndex:null,barycoord:null,object:a})}}class zi extends Ue{constructor(){super(),this.isGroup=!0,this.type="Group"}}class bc extends Pe{constructor(t,e,n,s,r,a,o,l,c){super(t,e,n,s,r,a,o,l,c),this.isCanvasTexture=!0,this.needsUpdate=!0}}class Tc extends Pe{constructor(t,e,n,s,r,a,o,l,c,h=Gi){if(h!==Gi&&h!==$i)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");n===void 0&&h===Gi&&(n=li),n===void 0&&h===$i&&(n=Ki),super(null,s,r,a,o,l,h,n,c),this.isDepthTexture=!0,this.image={width:t,height:e},this.magFilter=o!==void 0?o:Je,this.minFilter=l!==void 0?l:Je,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(t){return super.copy(t),this.compareFunction=t.compareFunction,this}toJSON(t){const e=super.toJSON(t);return this.compareFunction!==null&&(e.compareFunction=this.compareFunction),e}}class ws extends ge{constructor(t=1,e=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:t,height:e,widthSegments:n,heightSegments:s};const r=t/2,a=e/2,o=Math.floor(n),l=Math.floor(s),c=o+1,h=l+1,d=t/o,p=e/l,u=[],g=[],_=[],m=[];for(let f=0;f<h;f++){const T=f*p-a;for(let E=0;E<c;E++){const y=E*d-r;g.push(y,-T,0),_.push(0,0,1),m.push(E/o),m.push(1-f/l)}}for(let f=0;f<l;f++)for(let T=0;T<o;T++){const E=T+c*f,y=T+c*(f+1),D=T+1+c*(f+1),w=T+1+c*f;u.push(E,y,w),u.push(y,D,w)}this.setIndex(u),this.setAttribute("position",new Oe(g,3)),this.setAttribute("normal",new Oe(_,3)),this.setAttribute("uv",new Oe(m,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new ws(t.width,t.height,t.widthSegments,t.heightSegments)}}class Ro extends ge{constructor(t=.5,e=1,n=32,s=1,r=0,a=Math.PI*2){super(),this.type="RingGeometry",this.parameters={innerRadius:t,outerRadius:e,thetaSegments:n,phiSegments:s,thetaStart:r,thetaLength:a},n=Math.max(3,n),s=Math.max(1,s);const o=[],l=[],c=[],h=[];let d=t;const p=(e-t)/s,u=new P,g=new St;for(let _=0;_<=s;_++){for(let m=0;m<=n;m++){const f=r+m/n*a;u.x=d*Math.cos(f),u.y=d*Math.sin(f),l.push(u.x,u.y,u.z),c.push(0,0,1),g.x=(u.x/e+1)/2,g.y=(u.y/e+1)/2,h.push(g.x,g.y)}d+=p}for(let _=0;_<s;_++){const m=_*(n+1);for(let f=0;f<n;f++){const T=f+m,E=T,y=T+n+1,D=T+n+2,w=T+1;o.push(E,y,w),o.push(y,D,w)}}this.setIndex(o),this.setAttribute("position",new Oe(l,3)),this.setAttribute("normal",new Oe(c,3)),this.setAttribute("uv",new Oe(h,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new Ro(t.innerRadius,t.outerRadius,t.thetaSegments,t.phiSegments,t.thetaStart,t.thetaLength)}}class Lr extends ge{constructor(t=1,e=32,n=16,s=0,r=Math.PI*2,a=0,o=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:t,widthSegments:e,heightSegments:n,phiStart:s,phiLength:r,thetaStart:a,thetaLength:o},e=Math.max(3,Math.floor(e)),n=Math.max(2,Math.floor(n));const l=Math.min(a+o,Math.PI);let c=0;const h=[],d=new P,p=new P,u=[],g=[],_=[],m=[];for(let f=0;f<=n;f++){const T=[],E=f/n;let y=0;f===0&&a===0?y=.5/e:f===n&&l===Math.PI&&(y=-.5/e);for(let D=0;D<=e;D++){const w=D/e;d.x=-t*Math.cos(s+w*r)*Math.sin(a+E*o),d.y=t*Math.cos(a+E*o),d.z=t*Math.sin(s+w*r)*Math.sin(a+E*o),g.push(d.x,d.y,d.z),p.copy(d).normalize(),_.push(p.x,p.y,p.z),m.push(w+y,1-E),T.push(c++)}h.push(T)}for(let f=0;f<n;f++)for(let T=0;T<e;T++){const E=h[f][T+1],y=h[f][T],D=h[f+1][T],w=h[f+1][T+1];(f!==0||a>0)&&u.push(E,y,w),(f!==n-1||l<Math.PI)&&u.push(y,D,w)}this.setIndex(u),this.setAttribute("position",new Oe(g,3)),this.setAttribute("normal",new Oe(_,3)),this.setAttribute("uv",new Oe(m,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new Lr(t.radius,t.widthSegments,t.heightSegments,t.phiStart,t.phiLength,t.thetaStart,t.thetaLength)}}class mu extends qn{constructor(t){super(),this.isMeshStandardMaterial=!0,this.type="MeshStandardMaterial",this.defines={STANDARD:""},this.color=new ot(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new ot(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=dc,this.normalScale=new St(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Mn,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.defines={STANDARD:""},this.color.copy(t.color),this.roughness=t.roughness,this.metalness=t.metalness,this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.roughnessMap=t.roughnessMap,this.metalnessMap=t.metalnessMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.envMapIntensity=t.envMapIntensity,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this.fog=t.fog,this}}class gu extends qn{constructor(t){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=Th,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(t)}copy(t){return super.copy(t),this.depthPacking=t.depthPacking,this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this}}class _u extends qn{constructor(t){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(t)}copy(t){return super.copy(t),this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this}}class wc extends Ue{constructor(t,e=1){super(),this.isLight=!0,this.type="Light",this.color=new ot(t),this.intensity=e}dispose(){}copy(t,e){return super.copy(t,e),this.color.copy(t.color),this.intensity=t.intensity,this}toJSON(t){const e=super.toJSON(t);return e.object.color=this.color.getHex(),e.object.intensity=this.intensity,this.groundColor!==void 0&&(e.object.groundColor=this.groundColor.getHex()),this.distance!==void 0&&(e.object.distance=this.distance),this.angle!==void 0&&(e.object.angle=this.angle),this.decay!==void 0&&(e.object.decay=this.decay),this.penumbra!==void 0&&(e.object.penumbra=this.penumbra),this.shadow!==void 0&&(e.object.shadow=this.shadow.toJSON()),this.target!==void 0&&(e.object.target=this.target.uuid),e}}const la=new ne,cl=new P,hl=new P;class vu{constructor(t){this.camera=t,this.intensity=1,this.bias=0,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new St(512,512),this.map=null,this.mapPass=null,this.matrix=new ne,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new Ao,this._frameExtents=new St(1,1),this._viewportCount=1,this._viewports=[new oe(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(t){const e=this.camera,n=this.matrix;cl.setFromMatrixPosition(t.matrixWorld),e.position.copy(cl),hl.setFromMatrixPosition(t.target.matrixWorld),e.lookAt(hl),e.updateMatrixWorld(),la.multiplyMatrices(e.projectionMatrix,e.matrixWorldInverse),this._frustum.setFromProjectionMatrix(la),n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply(la)}getViewport(t){return this._viewports[t]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(t){return this.camera=t.camera.clone(),this.intensity=t.intensity,this.bias=t.bias,this.radius=t.radius,this.mapSize.copy(t.mapSize),this}clone(){return new this.constructor().copy(this)}toJSON(){const t={};return this.intensity!==1&&(t.intensity=this.intensity),this.bias!==0&&(t.bias=this.bias),this.normalBias!==0&&(t.normalBias=this.normalBias),this.radius!==1&&(t.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(t.mapSize=this.mapSize.toArray()),t.camera=this.camera.toJSON(!1).object,delete t.camera.matrix,t}}const ul=new ne,ps=new P,ca=new P;class xu extends vu{constructor(){super(new $e(90,1,.5,500)),this.isPointLightShadow=!0,this._frameExtents=new St(4,2),this._viewportCount=6,this._viewports=[new oe(2,1,1,1),new oe(0,1,1,1),new oe(3,1,1,1),new oe(1,1,1,1),new oe(3,0,1,1),new oe(1,0,1,1)],this._cubeDirections=[new P(1,0,0),new P(-1,0,0),new P(0,0,1),new P(0,0,-1),new P(0,1,0),new P(0,-1,0)],this._cubeUps=[new P(0,1,0),new P(0,1,0),new P(0,1,0),new P(0,1,0),new P(0,0,1),new P(0,0,-1)]}updateMatrices(t,e=0){const n=this.camera,s=this.matrix,r=t.distance||n.far;r!==n.far&&(n.far=r,n.updateProjectionMatrix()),ps.setFromMatrixPosition(t.matrixWorld),n.position.copy(ps),ca.copy(n.position),ca.add(this._cubeDirections[e]),n.up.copy(this._cubeUps[e]),n.lookAt(ca),n.updateMatrixWorld(),s.makeTranslation(-ps.x,-ps.y,-ps.z),ul.multiplyMatrices(n.projectionMatrix,n.matrixWorldInverse),this._frustum.setFromProjectionMatrix(ul)}}class dl extends wc{constructor(t,e,n=0,s=2){super(t,e),this.isPointLight=!0,this.type="PointLight",this.distance=n,this.decay=s,this.shadow=new xu}get power(){return this.intensity*4*Math.PI}set power(t){this.intensity=t/(4*Math.PI)}dispose(){this.shadow.dispose()}copy(t,e){return super.copy(t,e),this.distance=t.distance,this.decay=t.decay,this.shadow=t.shadow.clone(),this}}class Ac extends Sc{constructor(t=-1,e=1,n=1,s=-1,r=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=t,this.right=e,this.top=n,this.bottom=s,this.near=r,this.far=a,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.left=t.left,this.right=t.right,this.top=t.top,this.bottom=t.bottom,this.near=t.near,this.far=t.far,this.zoom=t.zoom,this.view=t.view===null?null:Object.assign({},t.view),this}setViewOffset(t,e,n,s,r,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const t=(this.right-this.left)/(2*this.zoom),e=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=n-t,a=n+t,o=s+e,l=s-e;if(this.view!==null&&this.view.enabled){const c=(this.right-this.left)/this.view.fullWidth/this.zoom,h=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,a=r+c*this.view.width,o-=h*this.view.offsetY,l=o-h*this.view.height}this.projectionMatrix.makeOrthographic(r,a,o,l,this.near,this.far,this.coordinateSystem),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){const e=super.toJSON(t);return e.object.zoom=this.zoom,e.object.left=this.left,e.object.right=this.right,e.object.top=this.top,e.object.bottom=this.bottom,e.object.near=this.near,e.object.far=this.far,this.view!==null&&(e.object.view=Object.assign({},this.view)),e}}class Mu extends wc{constructor(t,e){super(t,e),this.isAmbientLight=!0,this.type="AmbientLight"}}class Su extends $e{constructor(t=[]){super(),this.isArrayCamera=!0,this.cameras=t}}class yu{constructor(t=!0){this.autoStart=t,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1}start(){this.startTime=fl(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let t=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){const e=fl();t=(e-this.oldTime)/1e3,this.oldTime=e,this.elapsedTime+=t}return t}}function fl(){return performance.now()}const pl=new ne;class Eu{constructor(t,e,n=0,s=1/0){this.ray=new bs(t,e),this.near=n,this.far=s,this.camera=null,this.layers=new wo,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(t,e){this.ray.set(t,e)}setFromCamera(t,e){e.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(t.x,t.y,.5).unproject(e).sub(this.ray.origin).normalize(),this.camera=e):e.isOrthographicCamera?(this.ray.origin.set(t.x,t.y,(e.near+e.far)/(e.near-e.far)).unproject(e),this.ray.direction.set(0,0,-1).transformDirection(e.matrixWorld),this.camera=e):console.error("THREE.Raycaster: Unsupported camera type: "+e.type)}setFromXRController(t){return pl.identity().extractRotation(t.matrixWorld),this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(0,0,-1).applyMatrix4(pl),this}intersectObject(t,e=!0,n=[]){return uo(t,this,n,e),n.sort(ml),n}intersectObjects(t,e=!0,n=[]){for(let s=0,r=t.length;s<r;s++)uo(t[s],this,n,e);return n.sort(ml),n}}function ml(i,t){return i.distance-t.distance}function uo(i,t,e,n){let s=!0;if(i.layers.test(t.layers)&&i.raycast(t,e)===!1&&(s=!1),s===!0&&n===!0){const r=i.children;for(let a=0,o=r.length;a<o;a++)uo(r[a],t,e,!0)}}class gl{constructor(t=1,e=0,n=0){return this.radius=t,this.phi=e,this.theta=n,this}set(t,e,n){return this.radius=t,this.phi=e,this.theta=n,this}copy(t){return this.radius=t.radius,this.phi=t.phi,this.theta=t.theta,this}makeSafe(){return this.phi=Yt(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(t){return this.setFromCartesianCoords(t.x,t.y,t.z)}setFromCartesianCoords(t,e,n){return this.radius=Math.sqrt(t*t+e*e+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(t,n),this.phi=Math.acos(Yt(e/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}class bu extends hi{constructor(t,e=null){super(),this.object=t,this.domElement=e,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(){}disconnect(){}dispose(){}update(){}}function _l(i,t,e,n){const s=Tu(n);switch(e){case ac:return i*t;case lc:return i*t;case cc:return i*t*2;case yo:return i*t/s.components*s.byteLength;case Eo:return i*t/s.components*s.byteLength;case hc:return i*t*2/s.components*s.byteLength;case bo:return i*t*2/s.components*s.byteLength;case oc:return i*t*3/s.components*s.byteLength;case dn:return i*t*4/s.components*s.byteLength;case To:return i*t*4/s.components*s.byteLength;case ur:case dr:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*8;case fr:case pr:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case Fa:case Ba:return Math.max(i,16)*Math.max(t,8)/4;case Na:case Oa:return Math.max(i,8)*Math.max(t,8)/2;case za:case ka:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*8;case Ha:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case Va:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case Ga:return Math.floor((i+4)/5)*Math.floor((t+3)/4)*16;case Wa:return Math.floor((i+4)/5)*Math.floor((t+4)/5)*16;case Xa:return Math.floor((i+5)/6)*Math.floor((t+4)/5)*16;case Ya:return Math.floor((i+5)/6)*Math.floor((t+5)/6)*16;case qa:return Math.floor((i+7)/8)*Math.floor((t+4)/5)*16;case ja:return Math.floor((i+7)/8)*Math.floor((t+5)/6)*16;case Za:return Math.floor((i+7)/8)*Math.floor((t+7)/8)*16;case Ka:return Math.floor((i+9)/10)*Math.floor((t+4)/5)*16;case $a:return Math.floor((i+9)/10)*Math.floor((t+5)/6)*16;case Ja:return Math.floor((i+9)/10)*Math.floor((t+7)/8)*16;case Qa:return Math.floor((i+9)/10)*Math.floor((t+9)/10)*16;case to:return Math.floor((i+11)/12)*Math.floor((t+9)/10)*16;case eo:return Math.floor((i+11)/12)*Math.floor((t+11)/12)*16;case mr:case no:case io:return Math.ceil(i/4)*Math.ceil(t/4)*16;case uc:case so:return Math.ceil(i/4)*Math.ceil(t/4)*8;case ro:case ao:return Math.ceil(i/4)*Math.ceil(t/4)*16}throw new Error(`Unable to determine texture byte length for ${e} format.`)}function Tu(i){switch(i){case Ln:case ic:return{byteLength:1,components:1};case Ms:case sc:case Pn:return{byteLength:2,components:1};case Mo:case So:return{byteLength:2,components:4};case li:case xo:case xn:return{byteLength:4,components:1};case rc:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:vo}}));typeof window<"u"&&(window.__THREE__?console.warn("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=vo);/**
 * @license
 * Copyright 2010-2024 Three.js Authors
 * SPDX-License-Identifier: MIT
 */function Rc(){let i=null,t=!1,e=null,n=null;function s(r,a){e(r,a),n=i.requestAnimationFrame(s)}return{start:function(){t!==!0&&e!==null&&(n=i.requestAnimationFrame(s),t=!0)},stop:function(){i.cancelAnimationFrame(n),t=!1},setAnimationLoop:function(r){e=r},setContext:function(r){i=r}}}function wu(i){const t=new WeakMap;function e(o,l){const c=o.array,h=o.usage,d=c.byteLength,p=i.createBuffer();i.bindBuffer(l,p),i.bufferData(l,c,h),o.onUploadCallback();let u;if(c instanceof Float32Array)u=i.FLOAT;else if(c instanceof Uint16Array)o.isFloat16BufferAttribute?u=i.HALF_FLOAT:u=i.UNSIGNED_SHORT;else if(c instanceof Int16Array)u=i.SHORT;else if(c instanceof Uint32Array)u=i.UNSIGNED_INT;else if(c instanceof Int32Array)u=i.INT;else if(c instanceof Int8Array)u=i.BYTE;else if(c instanceof Uint8Array)u=i.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)u=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:p,type:u,bytesPerElement:c.BYTES_PER_ELEMENT,version:o.version,size:d}}function n(o,l,c){const h=l.array,d=l.updateRanges;if(i.bindBuffer(c,o),d.length===0)i.bufferSubData(c,0,h);else{d.sort((u,g)=>u.start-g.start);let p=0;for(let u=1;u<d.length;u++){const g=d[p],_=d[u];_.start<=g.start+g.count+1?g.count=Math.max(g.count,_.start+_.count-g.start):(++p,d[p]=_)}d.length=p+1;for(let u=0,g=d.length;u<g;u++){const _=d[u];i.bufferSubData(c,_.start*h.BYTES_PER_ELEMENT,h,_.start,_.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(o){return o.isInterleavedBufferAttribute&&(o=o.data),t.get(o)}function r(o){o.isInterleavedBufferAttribute&&(o=o.data);const l=t.get(o);l&&(i.deleteBuffer(l.buffer),t.delete(o))}function a(o,l){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){const h=t.get(o);(!h||h.version<o.version)&&t.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}const c=t.get(o);if(c===void 0)t.set(o,e(o,l));else if(c.version<o.version){if(c.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(c.buffer,o,l),c.version=o.version}}return{get:s,remove:r,update:a}}var Au=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,Ru=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,Cu=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,Pu=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Du=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,Lu=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Uu=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Iu=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Nu=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,Fu=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,Ou=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Bu=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,zu=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,ku=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Hu=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Vu=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,Gu=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Wu=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Xu=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Yu=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,qu=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,ju=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,Zu=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,Ku=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,$u=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,Ju=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,Qu=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,td=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,ed=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,nd=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,id="gl_FragColor = linearToOutputTexel( gl_FragColor );",sd=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,rd=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,ad=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif`,od=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,ld=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,cd=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,hd=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,ud=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,dd=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,fd=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,pd=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,md=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,gd=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,_d=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,vd=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,xd=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,Md=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Sd=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,yd=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Ed=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,bd=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,Td=`struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,wd=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,Ad=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,Rd=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Cd=`#if defined( USE_LOGDEPTHBUF )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Pd=`#if defined( USE_LOGDEPTHBUF )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Dd=`#ifdef USE_LOGDEPTHBUF
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Ld=`#ifdef USE_LOGDEPTHBUF
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Ud=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Id=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Nd=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Fd=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Od=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Bd=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,zd=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,kd=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Hd=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Vd=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,Gd=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Wd=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,Xd=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Yd=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,qd=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,jd=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,Zd=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,Kd=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,$d=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,Jd=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,Qd=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,tf=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,ef=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,nf=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,sf=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,rf=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,af=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,of=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,lf=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,cf=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		return step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow (sampler2D shadow, vec2 uv, float compare ){
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		float hard_shadow = step( compare , distribution.x );
		if (hard_shadow != 1.0 ) {
			float distance = compare - distribution.x ;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`,hf=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,uf=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,df=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,ff=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,pf=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,mf=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,gf=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,_f=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,vf=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,xf=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,Mf=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,Sf=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,yf=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,Ef=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,bf=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Tf=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,wf=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const Af=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,Rf=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Cf=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Pf=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Df=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Lf=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Uf=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,If=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	float fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5;
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Nf=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Ff=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,Of=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Bf=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,zf=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,kf=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Hf=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Vf=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Gf=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Wf=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Xf=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Yf=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,qf=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,jf=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Zf=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Kf=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,$f=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,Jf=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Qf=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,tp=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,ep=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,np=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,ip=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,sp=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,rp=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,ap=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,Gt={alphahash_fragment:Au,alphahash_pars_fragment:Ru,alphamap_fragment:Cu,alphamap_pars_fragment:Pu,alphatest_fragment:Du,alphatest_pars_fragment:Lu,aomap_fragment:Uu,aomap_pars_fragment:Iu,batching_pars_vertex:Nu,batching_vertex:Fu,begin_vertex:Ou,beginnormal_vertex:Bu,bsdfs:zu,iridescence_fragment:ku,bumpmap_pars_fragment:Hu,clipping_planes_fragment:Vu,clipping_planes_pars_fragment:Gu,clipping_planes_pars_vertex:Wu,clipping_planes_vertex:Xu,color_fragment:Yu,color_pars_fragment:qu,color_pars_vertex:ju,color_vertex:Zu,common:Ku,cube_uv_reflection_fragment:$u,defaultnormal_vertex:Ju,displacementmap_pars_vertex:Qu,displacementmap_vertex:td,emissivemap_fragment:ed,emissivemap_pars_fragment:nd,colorspace_fragment:id,colorspace_pars_fragment:sd,envmap_fragment:rd,envmap_common_pars_fragment:ad,envmap_pars_fragment:od,envmap_pars_vertex:ld,envmap_physical_pars_fragment:xd,envmap_vertex:cd,fog_vertex:hd,fog_pars_vertex:ud,fog_fragment:dd,fog_pars_fragment:fd,gradientmap_pars_fragment:pd,lightmap_pars_fragment:md,lights_lambert_fragment:gd,lights_lambert_pars_fragment:_d,lights_pars_begin:vd,lights_toon_fragment:Md,lights_toon_pars_fragment:Sd,lights_phong_fragment:yd,lights_phong_pars_fragment:Ed,lights_physical_fragment:bd,lights_physical_pars_fragment:Td,lights_fragment_begin:wd,lights_fragment_maps:Ad,lights_fragment_end:Rd,logdepthbuf_fragment:Cd,logdepthbuf_pars_fragment:Pd,logdepthbuf_pars_vertex:Dd,logdepthbuf_vertex:Ld,map_fragment:Ud,map_pars_fragment:Id,map_particle_fragment:Nd,map_particle_pars_fragment:Fd,metalnessmap_fragment:Od,metalnessmap_pars_fragment:Bd,morphinstance_vertex:zd,morphcolor_vertex:kd,morphnormal_vertex:Hd,morphtarget_pars_vertex:Vd,morphtarget_vertex:Gd,normal_fragment_begin:Wd,normal_fragment_maps:Xd,normal_pars_fragment:Yd,normal_pars_vertex:qd,normal_vertex:jd,normalmap_pars_fragment:Zd,clearcoat_normal_fragment_begin:Kd,clearcoat_normal_fragment_maps:$d,clearcoat_pars_fragment:Jd,iridescence_pars_fragment:Qd,opaque_fragment:tf,packing:ef,premultiplied_alpha_fragment:nf,project_vertex:sf,dithering_fragment:rf,dithering_pars_fragment:af,roughnessmap_fragment:of,roughnessmap_pars_fragment:lf,shadowmap_pars_fragment:cf,shadowmap_pars_vertex:hf,shadowmap_vertex:uf,shadowmask_pars_fragment:df,skinbase_vertex:ff,skinning_pars_vertex:pf,skinning_vertex:mf,skinnormal_vertex:gf,specularmap_fragment:_f,specularmap_pars_fragment:vf,tonemapping_fragment:xf,tonemapping_pars_fragment:Mf,transmission_fragment:Sf,transmission_pars_fragment:yf,uv_pars_fragment:Ef,uv_pars_vertex:bf,uv_vertex:Tf,worldpos_vertex:wf,background_vert:Af,background_frag:Rf,backgroundCube_vert:Cf,backgroundCube_frag:Pf,cube_vert:Df,cube_frag:Lf,depth_vert:Uf,depth_frag:If,distanceRGBA_vert:Nf,distanceRGBA_frag:Ff,equirect_vert:Of,equirect_frag:Bf,linedashed_vert:zf,linedashed_frag:kf,meshbasic_vert:Hf,meshbasic_frag:Vf,meshlambert_vert:Gf,meshlambert_frag:Wf,meshmatcap_vert:Xf,meshmatcap_frag:Yf,meshnormal_vert:qf,meshnormal_frag:jf,meshphong_vert:Zf,meshphong_frag:Kf,meshphysical_vert:$f,meshphysical_frag:Jf,meshtoon_vert:Qf,meshtoon_frag:tp,points_vert:ep,points_frag:np,shadow_vert:ip,shadow_frag:sp,sprite_vert:rp,sprite_frag:ap},ct={common:{diffuse:{value:new ot(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Ht},alphaMap:{value:null},alphaMapTransform:{value:new Ht},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Ht}},envmap:{envMap:{value:null},envMapRotation:{value:new Ht},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Ht}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Ht}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Ht},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Ht},normalScale:{value:new St(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Ht},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Ht}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Ht}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Ht}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new ot(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new ot(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Ht},alphaTest:{value:0},uvTransform:{value:new Ht}},sprite:{diffuse:{value:new ot(16777215)},opacity:{value:1},center:{value:new St(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Ht},alphaMap:{value:null},alphaMapTransform:{value:new Ht},alphaTest:{value:0}}},mn={basic:{uniforms:ke([ct.common,ct.specularmap,ct.envmap,ct.aomap,ct.lightmap,ct.fog]),vertexShader:Gt.meshbasic_vert,fragmentShader:Gt.meshbasic_frag},lambert:{uniforms:ke([ct.common,ct.specularmap,ct.envmap,ct.aomap,ct.lightmap,ct.emissivemap,ct.bumpmap,ct.normalmap,ct.displacementmap,ct.fog,ct.lights,{emissive:{value:new ot(0)}}]),vertexShader:Gt.meshlambert_vert,fragmentShader:Gt.meshlambert_frag},phong:{uniforms:ke([ct.common,ct.specularmap,ct.envmap,ct.aomap,ct.lightmap,ct.emissivemap,ct.bumpmap,ct.normalmap,ct.displacementmap,ct.fog,ct.lights,{emissive:{value:new ot(0)},specular:{value:new ot(1118481)},shininess:{value:30}}]),vertexShader:Gt.meshphong_vert,fragmentShader:Gt.meshphong_frag},standard:{uniforms:ke([ct.common,ct.envmap,ct.aomap,ct.lightmap,ct.emissivemap,ct.bumpmap,ct.normalmap,ct.displacementmap,ct.roughnessmap,ct.metalnessmap,ct.fog,ct.lights,{emissive:{value:new ot(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:Gt.meshphysical_vert,fragmentShader:Gt.meshphysical_frag},toon:{uniforms:ke([ct.common,ct.aomap,ct.lightmap,ct.emissivemap,ct.bumpmap,ct.normalmap,ct.displacementmap,ct.gradientmap,ct.fog,ct.lights,{emissive:{value:new ot(0)}}]),vertexShader:Gt.meshtoon_vert,fragmentShader:Gt.meshtoon_frag},matcap:{uniforms:ke([ct.common,ct.bumpmap,ct.normalmap,ct.displacementmap,ct.fog,{matcap:{value:null}}]),vertexShader:Gt.meshmatcap_vert,fragmentShader:Gt.meshmatcap_frag},points:{uniforms:ke([ct.points,ct.fog]),vertexShader:Gt.points_vert,fragmentShader:Gt.points_frag},dashed:{uniforms:ke([ct.common,ct.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:Gt.linedashed_vert,fragmentShader:Gt.linedashed_frag},depth:{uniforms:ke([ct.common,ct.displacementmap]),vertexShader:Gt.depth_vert,fragmentShader:Gt.depth_frag},normal:{uniforms:ke([ct.common,ct.bumpmap,ct.normalmap,ct.displacementmap,{opacity:{value:1}}]),vertexShader:Gt.meshnormal_vert,fragmentShader:Gt.meshnormal_frag},sprite:{uniforms:ke([ct.sprite,ct.fog]),vertexShader:Gt.sprite_vert,fragmentShader:Gt.sprite_frag},background:{uniforms:{uvTransform:{value:new Ht},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:Gt.background_vert,fragmentShader:Gt.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Ht}},vertexShader:Gt.backgroundCube_vert,fragmentShader:Gt.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:Gt.cube_vert,fragmentShader:Gt.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:Gt.equirect_vert,fragmentShader:Gt.equirect_frag},distanceRGBA:{uniforms:ke([ct.common,ct.displacementmap,{referencePosition:{value:new P},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:Gt.distanceRGBA_vert,fragmentShader:Gt.distanceRGBA_frag},shadow:{uniforms:ke([ct.lights,ct.fog,{color:{value:new ot(0)},opacity:{value:1}}]),vertexShader:Gt.shadow_vert,fragmentShader:Gt.shadow_frag}};mn.physical={uniforms:ke([mn.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Ht},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Ht},clearcoatNormalScale:{value:new St(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Ht},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Ht},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Ht},sheen:{value:0},sheenColor:{value:new ot(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Ht},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Ht},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Ht},transmissionSamplerSize:{value:new St},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Ht},attenuationDistance:{value:0},attenuationColor:{value:new ot(0)},specularColor:{value:new ot(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Ht},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Ht},anisotropyVector:{value:new St},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Ht}}]),vertexShader:Gt.meshphysical_vert,fragmentShader:Gt.meshphysical_frag};const ar={r:0,b:0,g:0},Qn=new Mn,op=new ne;function lp(i,t,e,n,s,r,a){const o=new ot(0);let l=r===!0?0:1,c,h,d=null,p=0,u=null;function g(E){let y=E.isScene===!0?E.background:null;return y&&y.isTexture&&(y=(E.backgroundBlurriness>0?e:t).get(y)),y}function _(E){let y=!1;const D=g(E);D===null?f(o,l):D&&D.isColor&&(f(D,1),y=!0);const w=i.xr.getEnvironmentBlendMode();w==="additive"?n.buffers.color.setClear(0,0,0,1,a):w==="alpha-blend"&&n.buffers.color.setClear(0,0,0,0,a),(i.autoClear||y)&&(n.buffers.depth.setTest(!0),n.buffers.depth.setMask(!0),n.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function m(E,y){const D=g(y);D&&(D.isCubeTexture||D.mapping===Pr)?(h===void 0&&(h=new be(new Ts(1,1,1),new He({name:"BackgroundCubeMaterial",uniforms:Qi(mn.backgroundCube.uniforms),vertexShader:mn.backgroundCube.vertexShader,fragmentShader:mn.backgroundCube.fragmentShader,side:Xe,depthTest:!1,depthWrite:!1,fog:!1})),h.geometry.deleteAttribute("normal"),h.geometry.deleteAttribute("uv"),h.onBeforeRender=function(w,C,I){this.matrixWorld.copyPosition(I.matrixWorld)},Object.defineProperty(h.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(h)),Qn.copy(y.backgroundRotation),Qn.x*=-1,Qn.y*=-1,Qn.z*=-1,D.isCubeTexture&&D.isRenderTargetTexture===!1&&(Qn.y*=-1,Qn.z*=-1),h.material.uniforms.envMap.value=D,h.material.uniforms.flipEnvMap.value=D.isCubeTexture&&D.isRenderTargetTexture===!1?-1:1,h.material.uniforms.backgroundBlurriness.value=y.backgroundBlurriness,h.material.uniforms.backgroundIntensity.value=y.backgroundIntensity,h.material.uniforms.backgroundRotation.value.setFromMatrix4(op.makeRotationFromEuler(Qn)),h.material.toneMapped=Qt.getTransfer(D.colorSpace)!==se,(d!==D||p!==D.version||u!==i.toneMapping)&&(h.material.needsUpdate=!0,d=D,p=D.version,u=i.toneMapping),h.layers.enableAll(),E.unshift(h,h.geometry,h.material,0,0,null)):D&&D.isTexture&&(c===void 0&&(c=new be(new ws(2,2),new He({name:"BackgroundMaterial",uniforms:Qi(mn.background.uniforms),vertexShader:mn.background.vertexShader,fragmentShader:mn.background.fragmentShader,side:Yn,depthTest:!1,depthWrite:!1,fog:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(c)),c.material.uniforms.t2D.value=D,c.material.uniforms.backgroundIntensity.value=y.backgroundIntensity,c.material.toneMapped=Qt.getTransfer(D.colorSpace)!==se,D.matrixAutoUpdate===!0&&D.updateMatrix(),c.material.uniforms.uvTransform.value.copy(D.matrix),(d!==D||p!==D.version||u!==i.toneMapping)&&(c.material.needsUpdate=!0,d=D,p=D.version,u=i.toneMapping),c.layers.enableAll(),E.unshift(c,c.geometry,c.material,0,0,null))}function f(E,y){E.getRGB(ar,Mc(i)),n.buffers.color.setClear(ar.r,ar.g,ar.b,y,a)}function T(){h!==void 0&&(h.geometry.dispose(),h.material.dispose()),c!==void 0&&(c.geometry.dispose(),c.material.dispose())}return{getClearColor:function(){return o},setClearColor:function(E,y=1){o.set(E),l=y,f(o,l)},getClearAlpha:function(){return l},setClearAlpha:function(E){l=E,f(o,l)},render:_,addToRenderList:m,dispose:T}}function cp(i,t){const e=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=p(null);let r=s,a=!1;function o(M,A,W,k,q){let Q=!1;const X=d(k,W,A);r!==X&&(r=X,c(r.object)),Q=u(M,k,W,q),Q&&g(M,k,W,q),q!==null&&t.update(q,i.ELEMENT_ARRAY_BUFFER),(Q||a)&&(a=!1,y(M,A,W,k),q!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,t.get(q).buffer))}function l(){return i.createVertexArray()}function c(M){return i.bindVertexArray(M)}function h(M){return i.deleteVertexArray(M)}function d(M,A,W){const k=W.wireframe===!0;let q=n[M.id];q===void 0&&(q={},n[M.id]=q);let Q=q[A.id];Q===void 0&&(Q={},q[A.id]=Q);let X=Q[k];return X===void 0&&(X=p(l()),Q[k]=X),X}function p(M){const A=[],W=[],k=[];for(let q=0;q<e;q++)A[q]=0,W[q]=0,k[q]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:A,enabledAttributes:W,attributeDivisors:k,object:M,attributes:{},index:null}}function u(M,A,W,k){const q=r.attributes,Q=A.attributes;let X=0;const tt=W.getAttributes();for(const H in tt)if(tt[H].location>=0){const gt=q[H];let Et=Q[H];if(Et===void 0&&(H==="instanceMatrix"&&M.instanceMatrix&&(Et=M.instanceMatrix),H==="instanceColor"&&M.instanceColor&&(Et=M.instanceColor)),gt===void 0||gt.attribute!==Et||Et&&gt.data!==Et.data)return!0;X++}return r.attributesNum!==X||r.index!==k}function g(M,A,W,k){const q={},Q=A.attributes;let X=0;const tt=W.getAttributes();for(const H in tt)if(tt[H].location>=0){let gt=Q[H];gt===void 0&&(H==="instanceMatrix"&&M.instanceMatrix&&(gt=M.instanceMatrix),H==="instanceColor"&&M.instanceColor&&(gt=M.instanceColor));const Et={};Et.attribute=gt,gt&&gt.data&&(Et.data=gt.data),q[H]=Et,X++}r.attributes=q,r.attributesNum=X,r.index=k}function _(){const M=r.newAttributes;for(let A=0,W=M.length;A<W;A++)M[A]=0}function m(M){f(M,0)}function f(M,A){const W=r.newAttributes,k=r.enabledAttributes,q=r.attributeDivisors;W[M]=1,k[M]===0&&(i.enableVertexAttribArray(M),k[M]=1),q[M]!==A&&(i.vertexAttribDivisor(M,A),q[M]=A)}function T(){const M=r.newAttributes,A=r.enabledAttributes;for(let W=0,k=A.length;W<k;W++)A[W]!==M[W]&&(i.disableVertexAttribArray(W),A[W]=0)}function E(M,A,W,k,q,Q,X){X===!0?i.vertexAttribIPointer(M,A,W,q,Q):i.vertexAttribPointer(M,A,W,k,q,Q)}function y(M,A,W,k){_();const q=k.attributes,Q=W.getAttributes(),X=A.defaultAttributeValues;for(const tt in Q){const H=Q[tt];if(H.location>=0){let st=q[tt];if(st===void 0&&(tt==="instanceMatrix"&&M.instanceMatrix&&(st=M.instanceMatrix),tt==="instanceColor"&&M.instanceColor&&(st=M.instanceColor)),st!==void 0){const gt=st.normalized,Et=st.itemSize,Ft=t.get(st);if(Ft===void 0)continue;const Xt=Ft.buffer,Y=Ft.type,nt=Ft.bytesPerElement,_t=Y===i.INT||Y===i.UNSIGNED_INT||st.gpuType===xo;if(st.isInterleavedBufferAttribute){const lt=st.data,Ct=lt.stride,Lt=st.offset;if(lt.isInstancedInterleavedBuffer){for(let Vt=0;Vt<H.locationSize;Vt++)f(H.location+Vt,lt.meshPerAttribute);M.isInstancedMesh!==!0&&k._maxInstanceCount===void 0&&(k._maxInstanceCount=lt.meshPerAttribute*lt.count)}else for(let Vt=0;Vt<H.locationSize;Vt++)m(H.location+Vt);i.bindBuffer(i.ARRAY_BUFFER,Xt);for(let Vt=0;Vt<H.locationSize;Vt++)E(H.location+Vt,Et/H.locationSize,Y,gt,Ct*nt,(Lt+Et/H.locationSize*Vt)*nt,_t)}else{if(st.isInstancedBufferAttribute){for(let lt=0;lt<H.locationSize;lt++)f(H.location+lt,st.meshPerAttribute);M.isInstancedMesh!==!0&&k._maxInstanceCount===void 0&&(k._maxInstanceCount=st.meshPerAttribute*st.count)}else for(let lt=0;lt<H.locationSize;lt++)m(H.location+lt);i.bindBuffer(i.ARRAY_BUFFER,Xt);for(let lt=0;lt<H.locationSize;lt++)E(H.location+lt,Et/H.locationSize,Y,gt,Et*nt,Et/H.locationSize*lt*nt,_t)}}else if(X!==void 0){const gt=X[tt];if(gt!==void 0)switch(gt.length){case 2:i.vertexAttrib2fv(H.location,gt);break;case 3:i.vertexAttrib3fv(H.location,gt);break;case 4:i.vertexAttrib4fv(H.location,gt);break;default:i.vertexAttrib1fv(H.location,gt)}}}}T()}function D(){I();for(const M in n){const A=n[M];for(const W in A){const k=A[W];for(const q in k)h(k[q].object),delete k[q];delete A[W]}delete n[M]}}function w(M){if(n[M.id]===void 0)return;const A=n[M.id];for(const W in A){const k=A[W];for(const q in k)h(k[q].object),delete k[q];delete A[W]}delete n[M.id]}function C(M){for(const A in n){const W=n[A];if(W[M.id]===void 0)continue;const k=W[M.id];for(const q in k)h(k[q].object),delete k[q];delete W[M.id]}}function I(){S(),a=!0,r!==s&&(r=s,c(r.object))}function S(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:o,reset:I,resetDefaultState:S,dispose:D,releaseStatesOfGeometry:w,releaseStatesOfProgram:C,initAttributes:_,enableAttribute:m,disableUnusedAttributes:T}}function hp(i,t,e){let n;function s(c){n=c}function r(c,h){i.drawArrays(n,c,h),e.update(h,n,1)}function a(c,h,d){d!==0&&(i.drawArraysInstanced(n,c,h,d),e.update(h,n,d))}function o(c,h,d){if(d===0)return;t.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,h,0,d);let u=0;for(let g=0;g<d;g++)u+=h[g];e.update(u,n,1)}function l(c,h,d,p){if(d===0)return;const u=t.get("WEBGL_multi_draw");if(u===null)for(let g=0;g<c.length;g++)a(c[g],h[g],p[g]);else{u.multiDrawArraysInstancedWEBGL(n,c,0,h,0,p,0,d);let g=0;for(let _=0;_<d;_++)g+=h[_]*p[_];e.update(g,n,1)}}this.setMode=s,this.render=r,this.renderInstances=a,this.renderMultiDraw=o,this.renderMultiDrawInstances=l}function up(i,t,e,n){let s;function r(){if(s!==void 0)return s;if(t.has("EXT_texture_filter_anisotropic")===!0){const C=t.get("EXT_texture_filter_anisotropic");s=i.getParameter(C.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function a(C){return!(C!==dn&&n.convert(C)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(C){const I=C===Pn&&(t.has("EXT_color_buffer_half_float")||t.has("EXT_color_buffer_float"));return!(C!==Ln&&n.convert(C)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&C!==xn&&!I)}function l(C){if(C==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";C="mediump"}return C==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=e.precision!==void 0?e.precision:"highp";const h=l(c);h!==c&&(console.warn("THREE.WebGLRenderer:",c,"not supported, using",h,"instead."),c=h);const d=e.logarithmicDepthBuffer===!0,p=e.reverseDepthBuffer===!0&&t.has("EXT_clip_control"),u=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),g=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),_=i.getParameter(i.MAX_TEXTURE_SIZE),m=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),f=i.getParameter(i.MAX_VERTEX_ATTRIBS),T=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),E=i.getParameter(i.MAX_VARYING_VECTORS),y=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),D=g>0,w=i.getParameter(i.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:a,textureTypeReadable:o,precision:c,logarithmicDepthBuffer:d,reverseDepthBuffer:p,maxTextures:u,maxVertexTextures:g,maxTextureSize:_,maxCubemapSize:m,maxAttributes:f,maxVertexUniforms:T,maxVaryings:E,maxFragmentUniforms:y,vertexTextures:D,maxSamples:w}}function dp(i){const t=this;let e=null,n=0,s=!1,r=!1;const a=new Hn,o=new Ht,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(d,p){const u=d.length!==0||p||n!==0||s;return s=p,n=d.length,u},this.beginShadows=function(){r=!0,h(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(d,p){e=h(d,p,0)},this.setState=function(d,p,u){const g=d.clippingPlanes,_=d.clipIntersection,m=d.clipShadows,f=i.get(d);if(!s||g===null||g.length===0||r&&!m)r?h(null):c();else{const T=r?0:n,E=T*4;let y=f.clippingState||null;l.value=y,y=h(g,p,E,u);for(let D=0;D!==E;++D)y[D]=e[D];f.clippingState=y,this.numIntersection=_?this.numPlanes:0,this.numPlanes+=T}};function c(){l.value!==e&&(l.value=e,l.needsUpdate=n>0),t.numPlanes=n,t.numIntersection=0}function h(d,p,u,g){const _=d!==null?d.length:0;let m=null;if(_!==0){if(m=l.value,g!==!0||m===null){const f=u+_*4,T=p.matrixWorldInverse;o.getNormalMatrix(T),(m===null||m.length<f)&&(m=new Float32Array(f));for(let E=0,y=u;E!==_;++E,y+=4)a.copy(d[E]).applyMatrix4(T,o),a.normal.toArray(m,y),m[y+3]=a.constant}l.value=m,l.needsUpdate=!0}return t.numPlanes=_,t.numIntersection=0,m}}function fp(i){let t=new WeakMap;function e(a,o){return o===Da?a.mapping=ji:o===La&&(a.mapping=Zi),a}function n(a){if(a&&a.isTexture){const o=a.mapping;if(o===Da||o===La)if(t.has(a)){const l=t.get(a).texture;return e(l,a.mapping)}else{const l=a.image;if(l&&l.height>0){const c=new ou(l.height);return c.fromEquirectangularTexture(i,a),t.set(a,c),a.addEventListener("dispose",s),e(c.texture,a.mapping)}else return null}}return a}function s(a){const o=a.target;o.removeEventListener("dispose",s);const l=t.get(o);l!==void 0&&(t.delete(o),l.dispose())}function r(){t=new WeakMap}return{get:n,dispose:r}}const ki=4,vl=[.125,.215,.35,.446,.526,.582],si=20,ha=new Ac,xl=new ot;let ua=null,da=0,fa=0,pa=!1;const ei=(1+Math.sqrt(5))/2,Ni=1/ei,Ml=[new P(-ei,Ni,0),new P(ei,Ni,0),new P(-Ni,0,ei),new P(Ni,0,ei),new P(0,ei,-Ni),new P(0,ei,Ni),new P(-1,1,-1),new P(1,1,-1),new P(-1,1,1),new P(1,1,1)];class Sl{constructor(t){this._renderer=t,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._lodPlanes=[],this._sizeLods=[],this._sigmas=[],this._blurMaterial=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._compileMaterial(this._blurMaterial)}fromScene(t,e=0,n=.1,s=100){ua=this._renderer.getRenderTarget(),da=this._renderer.getActiveCubeFace(),fa=this._renderer.getActiveMipmapLevel(),pa=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(256);const r=this._allocateTargets();return r.depthBuffer=!0,this._sceneToCubeUV(t,n,s,r),e>0&&this._blur(r,0,0,e),this._applyPMREM(r),this._cleanup(r),r}fromEquirectangular(t,e=null){return this._fromTexture(t,e)}fromCubemap(t,e=null){return this._fromTexture(t,e)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=bl(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=El(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose()}_setSize(t){this._lodMax=Math.floor(Math.log2(t)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let t=0;t<this._lodPlanes.length;t++)this._lodPlanes[t].dispose()}_cleanup(t){this._renderer.setRenderTarget(ua,da,fa),this._renderer.xr.enabled=pa,t.scissorTest=!1,or(t,0,0,t.width,t.height)}_fromTexture(t,e){t.mapping===ji||t.mapping===Zi?this._setSize(t.image.length===0?16:t.image[0].width||t.image[0].image.width):this._setSize(t.image.width/4),ua=this._renderer.getRenderTarget(),da=this._renderer.getActiveCubeFace(),fa=this._renderer.getActiveMipmapLevel(),pa=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=e||this._allocateTargets();return this._textureToCubeUV(t,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const t=3*Math.max(this._cubeSize,112),e=4*this._cubeSize,n={magFilter:vn,minFilter:vn,generateMipmaps:!1,type:Pn,format:dn,colorSpace:Ji,depthBuffer:!1},s=yl(t,e,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==t||this._pingPongRenderTarget.height!==e){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=yl(t,e,n);const{_lodMax:r}=this;({sizeLods:this._sizeLods,lodPlanes:this._lodPlanes,sigmas:this._sigmas}=pp(r)),this._blurMaterial=mp(r,t,e)}return s}_compileMaterial(t){const e=new be(this._lodPlanes[0],t);this._renderer.compile(e,ha)}_sceneToCubeUV(t,e,n,s){const o=new $e(90,1,e,n),l=[1,-1,1,1,1,1],c=[1,1,1,-1,-1,-1],h=this._renderer,d=h.autoClear,p=h.toneMapping;h.getClearColor(xl),h.toneMapping=Wn,h.autoClear=!1;const u=new Ss({name:"PMREM.Background",side:Xe,depthWrite:!1,depthTest:!1}),g=new be(new Ts,u);let _=!1;const m=t.background;m?m.isColor&&(u.color.copy(m),t.background=null,_=!0):(u.color.copy(xl),_=!0);for(let f=0;f<6;f++){const T=f%3;T===0?(o.up.set(0,l[f],0),o.lookAt(c[f],0,0)):T===1?(o.up.set(0,0,l[f]),o.lookAt(0,c[f],0)):(o.up.set(0,l[f],0),o.lookAt(0,0,c[f]));const E=this._cubeSize;or(s,T*E,f>2?E:0,E,E),h.setRenderTarget(s),_&&h.render(g,o),h.render(t,o)}g.geometry.dispose(),g.material.dispose(),h.toneMapping=p,h.autoClear=d,t.background=m}_textureToCubeUV(t,e){const n=this._renderer,s=t.mapping===ji||t.mapping===Zi;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=bl()),this._cubemapMaterial.uniforms.flipEnvMap.value=t.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=El());const r=s?this._cubemapMaterial:this._equirectMaterial,a=new be(this._lodPlanes[0],r),o=r.uniforms;o.envMap.value=t;const l=this._cubeSize;or(e,0,0,3*l,2*l),n.setRenderTarget(e),n.render(a,ha)}_applyPMREM(t){const e=this._renderer,n=e.autoClear;e.autoClear=!1;const s=this._lodPlanes.length;for(let r=1;r<s;r++){const a=Math.sqrt(this._sigmas[r]*this._sigmas[r]-this._sigmas[r-1]*this._sigmas[r-1]),o=Ml[(s-r-1)%Ml.length];this._blur(t,r-1,r,a,o)}e.autoClear=n}_blur(t,e,n,s,r){const a=this._pingPongRenderTarget;this._halfBlur(t,a,e,n,s,"latitudinal",r),this._halfBlur(a,t,n,n,s,"longitudinal",r)}_halfBlur(t,e,n,s,r,a,o){const l=this._renderer,c=this._blurMaterial;a!=="latitudinal"&&a!=="longitudinal"&&console.error("blur direction must be either latitudinal or longitudinal!");const h=3,d=new be(this._lodPlanes[s],c),p=c.uniforms,u=this._sizeLods[n]-1,g=isFinite(r)?Math.PI/(2*u):2*Math.PI/(2*si-1),_=r/g,m=isFinite(r)?1+Math.floor(h*_):si;m>si&&console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${si}`);const f=[];let T=0;for(let C=0;C<si;++C){const I=C/_,S=Math.exp(-I*I/2);f.push(S),C===0?T+=S:C<m&&(T+=2*S)}for(let C=0;C<f.length;C++)f[C]=f[C]/T;p.envMap.value=t.texture,p.samples.value=m,p.weights.value=f,p.latitudinal.value=a==="latitudinal",o&&(p.poleAxis.value=o);const{_lodMax:E}=this;p.dTheta.value=g,p.mipInt.value=E-n;const y=this._sizeLods[s],D=3*y*(s>E-ki?s-E+ki:0),w=4*(this._cubeSize-y);or(e,D,w,3*y,2*y),l.setRenderTarget(e),l.render(d,ha)}}function pp(i){const t=[],e=[],n=[];let s=i;const r=i-ki+1+vl.length;for(let a=0;a<r;a++){const o=Math.pow(2,s);e.push(o);let l=1/o;a>i-ki?l=vl[a-i+ki-1]:a===0&&(l=0),n.push(l);const c=1/(o-2),h=-c,d=1+c,p=[h,h,d,h,d,d,h,h,d,d,h,d],u=6,g=6,_=3,m=2,f=1,T=new Float32Array(_*g*u),E=new Float32Array(m*g*u),y=new Float32Array(f*g*u);for(let w=0;w<u;w++){const C=w%3*2/3-1,I=w>2?0:-1,S=[C,I,0,C+2/3,I,0,C+2/3,I+1,0,C,I,0,C+2/3,I+1,0,C,I+1,0];T.set(S,_*g*w),E.set(p,m*g*w);const M=[w,w,w,w,w,w];y.set(M,f*g*w)}const D=new ge;D.setAttribute("position",new he(T,_)),D.setAttribute("uv",new he(E,m)),D.setAttribute("faceIndex",new he(y,f)),t.push(D),s>ki&&s--}return{lodPlanes:t,sizeLods:e,sigmas:n}}function yl(i,t,e){const n=new fn(i,t,e);return n.texture.mapping=Pr,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function or(i,t,e,n,s){i.viewport.set(t,e,n,s),i.scissor.set(t,e,n,s)}function mp(i,t,e){const n=new Float32Array(si),s=new P(0,1,0);return new He({name:"SphericalGaussianBlur",defines:{n:si,CUBEUV_TEXEL_WIDTH:1/t,CUBEUV_TEXEL_HEIGHT:1/e,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Co(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:Cn,depthTest:!1,depthWrite:!1})}function El(){return new He({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Co(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:Cn,depthTest:!1,depthWrite:!1})}function bl(){return new He({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Co(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:Cn,depthTest:!1,depthWrite:!1})}function Co(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function gp(i){let t=new WeakMap,e=null;function n(o){if(o&&o.isTexture){const l=o.mapping,c=l===Da||l===La,h=l===ji||l===Zi;if(c||h){let d=t.get(o);const p=d!==void 0?d.texture.pmremVersion:0;if(o.isRenderTargetTexture&&o.pmremVersion!==p)return e===null&&(e=new Sl(i)),d=c?e.fromEquirectangular(o,d):e.fromCubemap(o,d),d.texture.pmremVersion=o.pmremVersion,t.set(o,d),d.texture;if(d!==void 0)return d.texture;{const u=o.image;return c&&u&&u.height>0||h&&u&&s(u)?(e===null&&(e=new Sl(i)),d=c?e.fromEquirectangular(o):e.fromCubemap(o),d.texture.pmremVersion=o.pmremVersion,t.set(o,d),o.addEventListener("dispose",r),d.texture):null}}}return o}function s(o){let l=0;const c=6;for(let h=0;h<c;h++)o[h]!==void 0&&l++;return l===c}function r(o){const l=o.target;l.removeEventListener("dispose",r);const c=t.get(l);c!==void 0&&(t.delete(l),c.dispose())}function a(){t=new WeakMap,e!==null&&(e.dispose(),e=null)}return{get:n,dispose:a}}function _p(i){const t={};function e(n){if(t[n]!==void 0)return t[n];let s;switch(n){case"WEBGL_depth_texture":s=i.getExtension("WEBGL_depth_texture")||i.getExtension("MOZ_WEBGL_depth_texture")||i.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=i.getExtension("EXT_texture_filter_anisotropic")||i.getExtension("MOZ_EXT_texture_filter_anisotropic")||i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=i.getExtension("WEBGL_compressed_texture_s3tc")||i.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=i.getExtension("WEBGL_compressed_texture_pvrtc")||i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=i.getExtension(n)}return t[n]=s,s}return{has:function(n){return e(n)!==null},init:function(){e("EXT_color_buffer_float"),e("WEBGL_clip_cull_distance"),e("OES_texture_float_linear"),e("EXT_color_buffer_half_float"),e("WEBGL_multisampled_render_to_texture"),e("WEBGL_render_shared_exponent")},get:function(n){const s=e(n);return s===null&&Fi("THREE.WebGLRenderer: "+n+" extension not supported."),s}}}function vp(i,t,e,n){const s={},r=new WeakMap;function a(d){const p=d.target;p.index!==null&&t.remove(p.index);for(const g in p.attributes)t.remove(p.attributes[g]);p.removeEventListener("dispose",a),delete s[p.id];const u=r.get(p);u&&(t.remove(u),r.delete(p)),n.releaseStatesOfGeometry(p),p.isInstancedBufferGeometry===!0&&delete p._maxInstanceCount,e.memory.geometries--}function o(d,p){return s[p.id]===!0||(p.addEventListener("dispose",a),s[p.id]=!0,e.memory.geometries++),p}function l(d){const p=d.attributes;for(const u in p)t.update(p[u],i.ARRAY_BUFFER)}function c(d){const p=[],u=d.index,g=d.attributes.position;let _=0;if(u!==null){const T=u.array;_=u.version;for(let E=0,y=T.length;E<y;E+=3){const D=T[E+0],w=T[E+1],C=T[E+2];p.push(D,w,w,C,C,D)}}else if(g!==void 0){const T=g.array;_=g.version;for(let E=0,y=T.length/3-1;E<y;E+=3){const D=E+0,w=E+1,C=E+2;p.push(D,w,w,C,C,D)}}else return;const m=new(pc(p)?xc:vc)(p,1);m.version=_;const f=r.get(d);f&&t.remove(f),r.set(d,m)}function h(d){const p=r.get(d);if(p){const u=d.index;u!==null&&p.version<u.version&&c(d)}else c(d);return r.get(d)}return{get:o,update:l,getWireframeAttribute:h}}function xp(i,t,e){let n;function s(p){n=p}let r,a;function o(p){r=p.type,a=p.bytesPerElement}function l(p,u){i.drawElements(n,u,r,p*a),e.update(u,n,1)}function c(p,u,g){g!==0&&(i.drawElementsInstanced(n,u,r,p*a,g),e.update(u,n,g))}function h(p,u,g){if(g===0)return;t.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,u,0,r,p,0,g);let m=0;for(let f=0;f<g;f++)m+=u[f];e.update(m,n,1)}function d(p,u,g,_){if(g===0)return;const m=t.get("WEBGL_multi_draw");if(m===null)for(let f=0;f<p.length;f++)c(p[f]/a,u[f],_[f]);else{m.multiDrawElementsInstancedWEBGL(n,u,0,r,p,0,_,0,g);let f=0;for(let T=0;T<g;T++)f+=u[T]*_[T];e.update(f,n,1)}}this.setMode=s,this.setIndex=o,this.render=l,this.renderInstances=c,this.renderMultiDraw=h,this.renderMultiDrawInstances=d}function Mp(i){const t={geometries:0,textures:0},e={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,a,o){switch(e.calls++,a){case i.TRIANGLES:e.triangles+=o*(r/3);break;case i.LINES:e.lines+=o*(r/2);break;case i.LINE_STRIP:e.lines+=o*(r-1);break;case i.LINE_LOOP:e.lines+=o*r;break;case i.POINTS:e.points+=o*r;break;default:console.error("THREE.WebGLInfo: Unknown draw mode:",a);break}}function s(){e.calls=0,e.triangles=0,e.points=0,e.lines=0}return{memory:t,render:e,programs:null,autoReset:!0,reset:s,update:n}}function Sp(i,t,e){const n=new WeakMap,s=new oe;function r(a,o,l){const c=a.morphTargetInfluences,h=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,d=h!==void 0?h.length:0;let p=n.get(o);if(p===void 0||p.count!==d){let S=function(){C.dispose(),n.delete(o),o.removeEventListener("dispose",S)};p!==void 0&&p.texture.dispose();const u=o.morphAttributes.position!==void 0,g=o.morphAttributes.normal!==void 0,_=o.morphAttributes.color!==void 0,m=o.morphAttributes.position||[],f=o.morphAttributes.normal||[],T=o.morphAttributes.color||[];let E=0;u===!0&&(E=1),g===!0&&(E=2),_===!0&&(E=3);let y=o.attributes.position.count*E,D=1;y>t.maxTextureSize&&(D=Math.ceil(y/t.maxTextureSize),y=t.maxTextureSize);const w=new Float32Array(y*D*4*d),C=new gc(w,y,D,d);C.type=xn,C.needsUpdate=!0;const I=E*4;for(let M=0;M<d;M++){const A=m[M],W=f[M],k=T[M],q=y*D*4*M;for(let Q=0;Q<A.count;Q++){const X=Q*I;u===!0&&(s.fromBufferAttribute(A,Q),w[q+X+0]=s.x,w[q+X+1]=s.y,w[q+X+2]=s.z,w[q+X+3]=0),g===!0&&(s.fromBufferAttribute(W,Q),w[q+X+4]=s.x,w[q+X+5]=s.y,w[q+X+6]=s.z,w[q+X+7]=0),_===!0&&(s.fromBufferAttribute(k,Q),w[q+X+8]=s.x,w[q+X+9]=s.y,w[q+X+10]=s.z,w[q+X+11]=k.itemSize===4?s.w:1)}}p={count:d,texture:C,size:new St(y,D)},n.set(o,p),o.addEventListener("dispose",S)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)l.getUniforms().setValue(i,"morphTexture",a.morphTexture,e);else{let u=0;for(let _=0;_<c.length;_++)u+=c[_];const g=o.morphTargetsRelative?1:1-u;l.getUniforms().setValue(i,"morphTargetBaseInfluence",g),l.getUniforms().setValue(i,"morphTargetInfluences",c)}l.getUniforms().setValue(i,"morphTargetsTexture",p.texture,e),l.getUniforms().setValue(i,"morphTargetsTextureSize",p.size)}return{update:r}}function yp(i,t,e,n){let s=new WeakMap;function r(l){const c=n.render.frame,h=l.geometry,d=t.get(l,h);if(s.get(d)!==c&&(t.update(d),s.set(d,c)),l.isInstancedMesh&&(l.hasEventListener("dispose",o)===!1&&l.addEventListener("dispose",o),s.get(l)!==c&&(e.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&e.update(l.instanceColor,i.ARRAY_BUFFER),s.set(l,c))),l.isSkinnedMesh){const p=l.skeleton;s.get(p)!==c&&(p.update(),s.set(p,c))}return d}function a(){s=new WeakMap}function o(l){const c=l.target;c.removeEventListener("dispose",o),e.remove(c.instanceMatrix),c.instanceColor!==null&&e.remove(c.instanceColor)}return{update:r,dispose:a}}const Cc=new Pe,Tl=new Tc(1,1),Pc=new gc,Dc=new Yh,Lc=new yc,wl=[],Al=[],Rl=new Float32Array(16),Cl=new Float32Array(9),Pl=new Float32Array(4);function es(i,t,e){const n=i[0];if(n<=0||n>0)return i;const s=t*e;let r=wl[s];if(r===void 0&&(r=new Float32Array(s),wl[s]=r),t!==0){n.toArray(r,0);for(let a=1,o=0;a!==t;++a)o+=e,i[a].toArray(r,o)}return r}function Te(i,t){if(i.length!==t.length)return!1;for(let e=0,n=i.length;e<n;e++)if(i[e]!==t[e])return!1;return!0}function we(i,t){for(let e=0,n=t.length;e<n;e++)i[e]=t[e]}function Ur(i,t){let e=Al[t];e===void 0&&(e=new Int32Array(t),Al[t]=e);for(let n=0;n!==t;++n)e[n]=i.allocateTextureUnit();return e}function Ep(i,t){const e=this.cache;e[0]!==t&&(i.uniform1f(this.addr,t),e[0]=t)}function bp(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2f(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Te(e,t))return;i.uniform2fv(this.addr,t),we(e,t)}}function Tp(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3f(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else if(t.r!==void 0)(e[0]!==t.r||e[1]!==t.g||e[2]!==t.b)&&(i.uniform3f(this.addr,t.r,t.g,t.b),e[0]=t.r,e[1]=t.g,e[2]=t.b);else{if(Te(e,t))return;i.uniform3fv(this.addr,t),we(e,t)}}function wp(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4f(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Te(e,t))return;i.uniform4fv(this.addr,t),we(e,t)}}function Ap(i,t){const e=this.cache,n=t.elements;if(n===void 0){if(Te(e,t))return;i.uniformMatrix2fv(this.addr,!1,t),we(e,t)}else{if(Te(e,n))return;Pl.set(n),i.uniformMatrix2fv(this.addr,!1,Pl),we(e,n)}}function Rp(i,t){const e=this.cache,n=t.elements;if(n===void 0){if(Te(e,t))return;i.uniformMatrix3fv(this.addr,!1,t),we(e,t)}else{if(Te(e,n))return;Cl.set(n),i.uniformMatrix3fv(this.addr,!1,Cl),we(e,n)}}function Cp(i,t){const e=this.cache,n=t.elements;if(n===void 0){if(Te(e,t))return;i.uniformMatrix4fv(this.addr,!1,t),we(e,t)}else{if(Te(e,n))return;Rl.set(n),i.uniformMatrix4fv(this.addr,!1,Rl),we(e,n)}}function Pp(i,t){const e=this.cache;e[0]!==t&&(i.uniform1i(this.addr,t),e[0]=t)}function Dp(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2i(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Te(e,t))return;i.uniform2iv(this.addr,t),we(e,t)}}function Lp(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3i(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if(Te(e,t))return;i.uniform3iv(this.addr,t),we(e,t)}}function Up(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4i(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Te(e,t))return;i.uniform4iv(this.addr,t),we(e,t)}}function Ip(i,t){const e=this.cache;e[0]!==t&&(i.uniform1ui(this.addr,t),e[0]=t)}function Np(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2ui(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Te(e,t))return;i.uniform2uiv(this.addr,t),we(e,t)}}function Fp(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3ui(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if(Te(e,t))return;i.uniform3uiv(this.addr,t),we(e,t)}}function Op(i,t){const e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4ui(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Te(e,t))return;i.uniform4uiv(this.addr,t),we(e,t)}}function Bp(i,t,e){const n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?(Tl.compareFunction=fc,r=Tl):r=Cc,e.setTexture2D(t||r,s)}function zp(i,t,e){const n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),e.setTexture3D(t||Dc,s)}function kp(i,t,e){const n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),e.setTextureCube(t||Lc,s)}function Hp(i,t,e){const n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),e.setTexture2DArray(t||Pc,s)}function Vp(i){switch(i){case 5126:return Ep;case 35664:return bp;case 35665:return Tp;case 35666:return wp;case 35674:return Ap;case 35675:return Rp;case 35676:return Cp;case 5124:case 35670:return Pp;case 35667:case 35671:return Dp;case 35668:case 35672:return Lp;case 35669:case 35673:return Up;case 5125:return Ip;case 36294:return Np;case 36295:return Fp;case 36296:return Op;case 35678:case 36198:case 36298:case 36306:case 35682:return Bp;case 35679:case 36299:case 36307:return zp;case 35680:case 36300:case 36308:case 36293:return kp;case 36289:case 36303:case 36311:case 36292:return Hp}}function Gp(i,t){i.uniform1fv(this.addr,t)}function Wp(i,t){const e=es(t,this.size,2);i.uniform2fv(this.addr,e)}function Xp(i,t){const e=es(t,this.size,3);i.uniform3fv(this.addr,e)}function Yp(i,t){const e=es(t,this.size,4);i.uniform4fv(this.addr,e)}function qp(i,t){const e=es(t,this.size,4);i.uniformMatrix2fv(this.addr,!1,e)}function jp(i,t){const e=es(t,this.size,9);i.uniformMatrix3fv(this.addr,!1,e)}function Zp(i,t){const e=es(t,this.size,16);i.uniformMatrix4fv(this.addr,!1,e)}function Kp(i,t){i.uniform1iv(this.addr,t)}function $p(i,t){i.uniform2iv(this.addr,t)}function Jp(i,t){i.uniform3iv(this.addr,t)}function Qp(i,t){i.uniform4iv(this.addr,t)}function tm(i,t){i.uniform1uiv(this.addr,t)}function em(i,t){i.uniform2uiv(this.addr,t)}function nm(i,t){i.uniform3uiv(this.addr,t)}function im(i,t){i.uniform4uiv(this.addr,t)}function sm(i,t,e){const n=this.cache,s=t.length,r=Ur(e,s);Te(n,r)||(i.uniform1iv(this.addr,r),we(n,r));for(let a=0;a!==s;++a)e.setTexture2D(t[a]||Cc,r[a])}function rm(i,t,e){const n=this.cache,s=t.length,r=Ur(e,s);Te(n,r)||(i.uniform1iv(this.addr,r),we(n,r));for(let a=0;a!==s;++a)e.setTexture3D(t[a]||Dc,r[a])}function am(i,t,e){const n=this.cache,s=t.length,r=Ur(e,s);Te(n,r)||(i.uniform1iv(this.addr,r),we(n,r));for(let a=0;a!==s;++a)e.setTextureCube(t[a]||Lc,r[a])}function om(i,t,e){const n=this.cache,s=t.length,r=Ur(e,s);Te(n,r)||(i.uniform1iv(this.addr,r),we(n,r));for(let a=0;a!==s;++a)e.setTexture2DArray(t[a]||Pc,r[a])}function lm(i){switch(i){case 5126:return Gp;case 35664:return Wp;case 35665:return Xp;case 35666:return Yp;case 35674:return qp;case 35675:return jp;case 35676:return Zp;case 5124:case 35670:return Kp;case 35667:case 35671:return $p;case 35668:case 35672:return Jp;case 35669:case 35673:return Qp;case 5125:return tm;case 36294:return em;case 36295:return nm;case 36296:return im;case 35678:case 36198:case 36298:case 36306:case 35682:return sm;case 35679:case 36299:case 36307:return rm;case 35680:case 36300:case 36308:case 36293:return am;case 36289:case 36303:case 36311:case 36292:return om}}class cm{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.type=e.type,this.setValue=Vp(e.type)}}class hm{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.type=e.type,this.size=e.size,this.setValue=lm(e.type)}}class um{constructor(t){this.id=t,this.seq=[],this.map={}}setValue(t,e,n){const s=this.seq;for(let r=0,a=s.length;r!==a;++r){const o=s[r];o.setValue(t,e[o.id],n)}}}const ma=/(\w+)(\])?(\[|\.)?/g;function Dl(i,t){i.seq.push(t),i.map[t.id]=t}function dm(i,t,e){const n=i.name,s=n.length;for(ma.lastIndex=0;;){const r=ma.exec(n),a=ma.lastIndex;let o=r[1];const l=r[2]==="]",c=r[3];if(l&&(o=o|0),c===void 0||c==="["&&a+2===s){Dl(e,c===void 0?new cm(o,i,t):new hm(o,i,t));break}else{let d=e.map[o];d===void 0&&(d=new um(o),Dl(e,d)),e=d}}}class _r{constructor(t,e){this.seq=[],this.map={};const n=t.getProgramParameter(e,t.ACTIVE_UNIFORMS);for(let s=0;s<n;++s){const r=t.getActiveUniform(e,s),a=t.getUniformLocation(e,r.name);dm(r,a,this)}}setValue(t,e,n,s){const r=this.map[e];r!==void 0&&r.setValue(t,n,s)}setOptional(t,e,n){const s=e[n];s!==void 0&&this.setValue(t,n,s)}static upload(t,e,n,s){for(let r=0,a=e.length;r!==a;++r){const o=e[r],l=n[o.id];l.needsUpdate!==!1&&o.setValue(t,l.value,s)}}static seqWithValue(t,e){const n=[];for(let s=0,r=t.length;s!==r;++s){const a=t[s];a.id in e&&n.push(a)}return n}}function Ll(i,t,e){const n=i.createShader(t);return i.shaderSource(n,e),i.compileShader(n),n}const fm=37297;let pm=0;function mm(i,t){const e=i.split(`
`),n=[],s=Math.max(t-6,0),r=Math.min(t+6,e.length);for(let a=s;a<r;a++){const o=a+1;n.push(`${o===t?">":" "} ${o}: ${e[a]}`)}return n.join(`
`)}const Ul=new Ht;function gm(i){Qt._getMatrix(Ul,Qt.workingColorSpace,i);const t=`mat3( ${Ul.elements.map(e=>e.toFixed(4))} )`;switch(Qt.getTransfer(i)){case yr:return[t,"LinearTransferOETF"];case se:return[t,"sRGBTransferOETF"];default:return console.warn("THREE.WebGLProgram: Unsupported color space: ",i),[t,"LinearTransferOETF"]}}function Il(i,t,e){const n=i.getShaderParameter(t,i.COMPILE_STATUS),s=i.getShaderInfoLog(t).trim();if(n&&s==="")return"";const r=/ERROR: 0:(\d+)/.exec(s);if(r){const a=parseInt(r[1]);return e.toUpperCase()+`

`+s+`

`+mm(i.getShaderSource(t),a)}else return s}function _m(i,t){const e=gm(t);return[`vec4 ${i}( vec4 value ) {`,`	return ${e[1]}( vec4( value.rgb * ${e[0]}, value.a ) );`,"}"].join(`
`)}function vm(i,t){let e;switch(t){case vh:e="Linear";break;case xh:e="Reinhard";break;case Mh:e="Cineon";break;case ec:e="ACESFilmic";break;case yh:e="AgX";break;case Eh:e="Neutral";break;case Sh:e="Custom";break;default:console.warn("THREE.WebGLProgram: Unsupported toneMapping:",t),e="Linear"}return"vec3 "+i+"( vec3 color ) { return "+e+"ToneMapping( color ); }"}const lr=new P;function xm(){Qt.getLuminanceCoefficients(lr);const i=lr.x.toFixed(4),t=lr.y.toFixed(4),e=lr.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${t}, ${e} );`,"	return dot( weights, rgb );","}"].join(`
`)}function Mm(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(gs).join(`
`)}function Sm(i){const t=[];for(const e in i){const n=i[e];n!==!1&&t.push("#define "+e+" "+n)}return t.join(`
`)}function ym(i,t){const e={},n=i.getProgramParameter(t,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){const r=i.getActiveAttrib(t,s),a=r.name;let o=1;r.type===i.FLOAT_MAT2&&(o=2),r.type===i.FLOAT_MAT3&&(o=3),r.type===i.FLOAT_MAT4&&(o=4),e[a]={type:r.type,location:i.getAttribLocation(t,a),locationSize:o}}return e}function gs(i){return i!==""}function Nl(i,t){const e=t.numSpotLightShadows+t.numSpotLightMaps-t.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,t.numDirLights).replace(/NUM_SPOT_LIGHTS/g,t.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,t.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,e).replace(/NUM_RECT_AREA_LIGHTS/g,t.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,t.numPointLights).replace(/NUM_HEMI_LIGHTS/g,t.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,t.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,t.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,t.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,t.numPointLightShadows)}function Fl(i,t){return i.replace(/NUM_CLIPPING_PLANES/g,t.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,t.numClippingPlanes-t.numClipIntersection)}const Em=/^[ \t]*#include +<([\w\d./]+)>/gm;function fo(i){return i.replace(Em,Tm)}const bm=new Map;function Tm(i,t){let e=Gt[t];if(e===void 0){const n=bm.get(t);if(n!==void 0)e=Gt[n],console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',t,n);else throw new Error("Can not resolve #include <"+t+">")}return fo(e)}const wm=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Ol(i){return i.replace(wm,Am)}function Am(i,t,e,n){let s="";for(let r=parseInt(t);r<parseInt(e);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Bl(i){let t=`precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;return i.precision==="highp"?t+=`
#define HIGH_PRECISION`:i.precision==="mediump"?t+=`
#define MEDIUM_PRECISION`:i.precision==="lowp"&&(t+=`
#define LOW_PRECISION`),t}function Rm(i){let t="SHADOWMAP_TYPE_BASIC";return i.shadowMapType===Ql?t="SHADOWMAP_TYPE_PCF":i.shadowMapType===$c?t="SHADOWMAP_TYPE_PCF_SOFT":i.shadowMapType===An&&(t="SHADOWMAP_TYPE_VSM"),t}function Cm(i){let t="ENVMAP_TYPE_CUBE";if(i.envMap)switch(i.envMapMode){case ji:case Zi:t="ENVMAP_TYPE_CUBE";break;case Pr:t="ENVMAP_TYPE_CUBE_UV";break}return t}function Pm(i){let t="ENVMAP_MODE_REFLECTION";if(i.envMap)switch(i.envMapMode){case Zi:t="ENVMAP_MODE_REFRACTION";break}return t}function Dm(i){let t="ENVMAP_BLENDING_NONE";if(i.envMap)switch(i.combine){case tc:t="ENVMAP_BLENDING_MULTIPLY";break;case gh:t="ENVMAP_BLENDING_MIX";break;case _h:t="ENVMAP_BLENDING_ADD";break}return t}function Lm(i){const t=i.envMapCubeUVHeight;if(t===null)return null;const e=Math.log2(t)-2,n=1/t;return{texelWidth:1/(3*Math.max(Math.pow(2,e),112)),texelHeight:n,maxMip:e}}function Um(i,t,e,n){const s=i.getContext(),r=e.defines;let a=e.vertexShader,o=e.fragmentShader;const l=Rm(e),c=Cm(e),h=Pm(e),d=Dm(e),p=Lm(e),u=Mm(e),g=Sm(r),_=s.createProgram();let m,f,T=e.glslVersion?"#version "+e.glslVersion+`
`:"";e.isRawShaderMaterial?(m=["#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g].filter(gs).join(`
`),m.length>0&&(m+=`
`),f=["#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g].filter(gs).join(`
`),f.length>0&&(f+=`
`)):(m=[Bl(e),"#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g,e.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",e.batching?"#define USE_BATCHING":"",e.batchingColor?"#define USE_BATCHING_COLOR":"",e.instancing?"#define USE_INSTANCING":"",e.instancingColor?"#define USE_INSTANCING_COLOR":"",e.instancingMorph?"#define USE_INSTANCING_MORPH":"",e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.map?"#define USE_MAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+h:"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.displacementMap?"#define USE_DISPLACEMENTMAP":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.anisotropy?"#define USE_ANISOTROPY":"",e.anisotropyMap?"#define USE_ANISOTROPYMAP":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaHash?"#define USE_ALPHAHASH":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.mapUv?"#define MAP_UV "+e.mapUv:"",e.alphaMapUv?"#define ALPHAMAP_UV "+e.alphaMapUv:"",e.lightMapUv?"#define LIGHTMAP_UV "+e.lightMapUv:"",e.aoMapUv?"#define AOMAP_UV "+e.aoMapUv:"",e.emissiveMapUv?"#define EMISSIVEMAP_UV "+e.emissiveMapUv:"",e.bumpMapUv?"#define BUMPMAP_UV "+e.bumpMapUv:"",e.normalMapUv?"#define NORMALMAP_UV "+e.normalMapUv:"",e.displacementMapUv?"#define DISPLACEMENTMAP_UV "+e.displacementMapUv:"",e.metalnessMapUv?"#define METALNESSMAP_UV "+e.metalnessMapUv:"",e.roughnessMapUv?"#define ROUGHNESSMAP_UV "+e.roughnessMapUv:"",e.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+e.anisotropyMapUv:"",e.clearcoatMapUv?"#define CLEARCOATMAP_UV "+e.clearcoatMapUv:"",e.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+e.clearcoatNormalMapUv:"",e.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+e.clearcoatRoughnessMapUv:"",e.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+e.iridescenceMapUv:"",e.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+e.iridescenceThicknessMapUv:"",e.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+e.sheenColorMapUv:"",e.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+e.sheenRoughnessMapUv:"",e.specularMapUv?"#define SPECULARMAP_UV "+e.specularMapUv:"",e.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+e.specularColorMapUv:"",e.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+e.specularIntensityMapUv:"",e.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+e.transmissionMapUv:"",e.thicknessMapUv?"#define THICKNESSMAP_UV "+e.thicknessMapUv:"",e.vertexTangents&&e.flatShading===!1?"#define USE_TANGENT":"",e.vertexColors?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUv1s?"#define USE_UV1":"",e.vertexUv2s?"#define USE_UV2":"",e.vertexUv3s?"#define USE_UV3":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.flatShading?"#define FLAT_SHADED":"",e.skinning?"#define USE_SKINNING":"",e.morphTargets?"#define USE_MORPHTARGETS":"",e.morphNormals&&e.flatShading===!1?"#define USE_MORPHNORMALS":"",e.morphColors?"#define USE_MORPHCOLORS":"",e.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+e.morphTextureStride:"",e.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+e.morphTargetsCount:"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+l:"",e.sizeAttenuation?"#define USE_SIZEATTENUATION":"",e.numLightProbes>0?"#define USE_LIGHT_PROBES":"",e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",e.reverseDepthBuffer?"#define USE_REVERSEDEPTHBUF":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(gs).join(`
`),f=[Bl(e),"#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,g,e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",e.map?"#define USE_MAP":"",e.matcap?"#define USE_MATCAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+c:"",e.envMap?"#define "+h:"",e.envMap?"#define "+d:"",p?"#define CUBEUV_TEXEL_WIDTH "+p.texelWidth:"",p?"#define CUBEUV_TEXEL_HEIGHT "+p.texelHeight:"",p?"#define CUBEUV_MAX_MIP "+p.maxMip+".0":"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.anisotropy?"#define USE_ANISOTROPY":"",e.anisotropyMap?"#define USE_ANISOTROPYMAP":"",e.clearcoat?"#define USE_CLEARCOAT":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.dispersion?"#define USE_DISPERSION":"",e.iridescence?"#define USE_IRIDESCENCE":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaTest?"#define USE_ALPHATEST":"",e.alphaHash?"#define USE_ALPHAHASH":"",e.sheen?"#define USE_SHEEN":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.vertexTangents&&e.flatShading===!1?"#define USE_TANGENT":"",e.vertexColors||e.instancingColor||e.batchingColor?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUv1s?"#define USE_UV1":"",e.vertexUv2s?"#define USE_UV2":"",e.vertexUv3s?"#define USE_UV3":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.gradientMap?"#define USE_GRADIENTMAP":"",e.flatShading?"#define FLAT_SHADED":"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+l:"",e.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",e.numLightProbes>0?"#define USE_LIGHT_PROBES":"",e.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",e.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",e.logarithmicDepthBuffer?"#define USE_LOGDEPTHBUF":"",e.reverseDepthBuffer?"#define USE_REVERSEDEPTHBUF":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",e.toneMapping!==Wn?"#define TONE_MAPPING":"",e.toneMapping!==Wn?Gt.tonemapping_pars_fragment:"",e.toneMapping!==Wn?vm("toneMapping",e.toneMapping):"",e.dithering?"#define DITHERING":"",e.opaque?"#define OPAQUE":"",Gt.colorspace_pars_fragment,_m("linearToOutputTexel",e.outputColorSpace),xm(),e.useDepthPacking?"#define DEPTH_PACKING "+e.depthPacking:"",`
`].filter(gs).join(`
`)),a=fo(a),a=Nl(a,e),a=Fl(a,e),o=fo(o),o=Nl(o,e),o=Fl(o,e),a=Ol(a),o=Ol(o),e.isRawShaderMaterial!==!0&&(T=`#version 300 es
`,m=[u,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+m,f=["#define varying in",e.glslVersion===Fo?"":"layout(location = 0) out highp vec4 pc_fragColor;",e.glslVersion===Fo?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+f);const E=T+m+a,y=T+f+o,D=Ll(s,s.VERTEX_SHADER,E),w=Ll(s,s.FRAGMENT_SHADER,y);s.attachShader(_,D),s.attachShader(_,w),e.index0AttributeName!==void 0?s.bindAttribLocation(_,0,e.index0AttributeName):e.morphTargets===!0&&s.bindAttribLocation(_,0,"position"),s.linkProgram(_);function C(A){if(i.debug.checkShaderErrors){const W=s.getProgramInfoLog(_).trim(),k=s.getShaderInfoLog(D).trim(),q=s.getShaderInfoLog(w).trim();let Q=!0,X=!0;if(s.getProgramParameter(_,s.LINK_STATUS)===!1)if(Q=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,_,D,w);else{const tt=Il(s,D,"vertex"),H=Il(s,w,"fragment");console.error("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(_,s.VALIDATE_STATUS)+`

Material Name: `+A.name+`
Material Type: `+A.type+`

Program Info Log: `+W+`
`+tt+`
`+H)}else W!==""?console.warn("THREE.WebGLProgram: Program Info Log:",W):(k===""||q==="")&&(X=!1);X&&(A.diagnostics={runnable:Q,programLog:W,vertexShader:{log:k,prefix:m},fragmentShader:{log:q,prefix:f}})}s.deleteShader(D),s.deleteShader(w),I=new _r(s,_),S=ym(s,_)}let I;this.getUniforms=function(){return I===void 0&&C(this),I};let S;this.getAttributes=function(){return S===void 0&&C(this),S};let M=e.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return M===!1&&(M=s.getProgramParameter(_,fm)),M},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(_),this.program=void 0},this.type=e.shaderType,this.name=e.shaderName,this.id=pm++,this.cacheKey=t,this.usedTimes=1,this.program=_,this.vertexShader=D,this.fragmentShader=w,this}let Im=0;class Nm{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(t){const e=t.vertexShader,n=t.fragmentShader,s=this._getShaderStage(e),r=this._getShaderStage(n),a=this._getShaderCacheForMaterial(t);return a.has(s)===!1&&(a.add(s),s.usedTimes++),a.has(r)===!1&&(a.add(r),r.usedTimes++),this}remove(t){const e=this.materialCache.get(t);for(const n of e)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(t),this}getVertexShaderID(t){return this._getShaderStage(t.vertexShader).id}getFragmentShaderID(t){return this._getShaderStage(t.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(t){const e=this.materialCache;let n=e.get(t);return n===void 0&&(n=new Set,e.set(t,n)),n}_getShaderStage(t){const e=this.shaderCache;let n=e.get(t);return n===void 0&&(n=new Fm(t),e.set(t,n)),n}}class Fm{constructor(t){this.id=Im++,this.code=t,this.usedTimes=0}}function Om(i,t,e,n,s,r,a){const o=new wo,l=new Nm,c=new Set,h=[],d=s.logarithmicDepthBuffer,p=s.vertexTextures;let u=s.precision;const g={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function _(S){return c.add(S),S===0?"uv":`uv${S}`}function m(S,M,A,W,k){const q=W.fog,Q=k.geometry,X=S.isMeshStandardMaterial?W.environment:null,tt=(S.isMeshStandardMaterial?e:t).get(S.envMap||X),H=tt&&tt.mapping===Pr?tt.image.height:null,st=g[S.type];S.precision!==null&&(u=s.getMaxPrecision(S.precision),u!==S.precision&&console.warn("THREE.WebGLProgram.getParameters:",S.precision,"not supported, using",u,"instead."));const gt=Q.morphAttributes.position||Q.morphAttributes.normal||Q.morphAttributes.color,Et=gt!==void 0?gt.length:0;let Ft=0;Q.morphAttributes.position!==void 0&&(Ft=1),Q.morphAttributes.normal!==void 0&&(Ft=2),Q.morphAttributes.color!==void 0&&(Ft=3);let Xt,Y,nt,_t;if(st){const $t=mn[st];Xt=$t.vertexShader,Y=$t.fragmentShader}else Xt=S.vertexShader,Y=S.fragmentShader,l.update(S),nt=l.getVertexShaderID(S),_t=l.getFragmentShaderID(S);const lt=i.getRenderTarget(),Ct=i.state.buffers.depth.getReversed(),Lt=k.isInstancedMesh===!0,Vt=k.isBatchedMesh===!0,ie=!!S.map,Wt=!!S.matcap,ue=!!tt,R=!!S.aoMap,ye=!!S.lightMap,qt=!!S.bumpMap,jt=!!S.normalMap,At=!!S.displacementMap,le=!!S.emissiveMap,Rt=!!S.metalnessMap,b=!!S.roughnessMap,v=S.anisotropy>0,F=S.clearcoat>0,K=S.dispersion>0,J=S.iridescence>0,j=S.sheen>0,Tt=S.transmission>0,dt=v&&!!S.anisotropyMap,V=F&&!!S.clearcoatMap,at=F&&!!S.clearcoatNormalMap,Z=F&&!!S.clearcoatRoughnessMap,it=J&&!!S.iridescenceMap,pt=J&&!!S.iridescenceThicknessMap,wt=j&&!!S.sheenColorMap,ht=j&&!!S.sheenRoughnessMap,Bt=!!S.specularMap,Ut=!!S.specularColorMap,Zt=!!S.specularIntensityMap,L=Tt&&!!S.transmissionMap,rt=Tt&&!!S.thicknessMap,G=!!S.gradientMap,$=!!S.alphaMap,ft=S.alphaTest>0,ut=!!S.alphaHash,Ot=!!S.extensions;let ce=Wn;S.toneMapped&&(lt===null||lt.isXRRenderTarget===!0)&&(ce=i.toneMapping);const Ae={shaderID:st,shaderType:S.type,shaderName:S.name,vertexShader:Xt,fragmentShader:Y,defines:S.defines,customVertexShaderID:nt,customFragmentShaderID:_t,isRawShaderMaterial:S.isRawShaderMaterial===!0,glslVersion:S.glslVersion,precision:u,batching:Vt,batchingColor:Vt&&k._colorsTexture!==null,instancing:Lt,instancingColor:Lt&&k.instanceColor!==null,instancingMorph:Lt&&k.morphTexture!==null,supportsVertexTextures:p,outputColorSpace:lt===null?i.outputColorSpace:lt.isXRRenderTarget===!0?lt.texture.colorSpace:Ji,alphaToCoverage:!!S.alphaToCoverage,map:ie,matcap:Wt,envMap:ue,envMapMode:ue&&tt.mapping,envMapCubeUVHeight:H,aoMap:R,lightMap:ye,bumpMap:qt,normalMap:jt,displacementMap:p&&At,emissiveMap:le,normalMapObjectSpace:jt&&S.normalMapType===Ah,normalMapTangentSpace:jt&&S.normalMapType===dc,metalnessMap:Rt,roughnessMap:b,anisotropy:v,anisotropyMap:dt,clearcoat:F,clearcoatMap:V,clearcoatNormalMap:at,clearcoatRoughnessMap:Z,dispersion:K,iridescence:J,iridescenceMap:it,iridescenceThicknessMap:pt,sheen:j,sheenColorMap:wt,sheenRoughnessMap:ht,specularMap:Bt,specularColorMap:Ut,specularIntensityMap:Zt,transmission:Tt,transmissionMap:L,thicknessMap:rt,gradientMap:G,opaque:S.transparent===!1&&S.blending===Vi&&S.alphaToCoverage===!1,alphaMap:$,alphaTest:ft,alphaHash:ut,combine:S.combine,mapUv:ie&&_(S.map.channel),aoMapUv:R&&_(S.aoMap.channel),lightMapUv:ye&&_(S.lightMap.channel),bumpMapUv:qt&&_(S.bumpMap.channel),normalMapUv:jt&&_(S.normalMap.channel),displacementMapUv:At&&_(S.displacementMap.channel),emissiveMapUv:le&&_(S.emissiveMap.channel),metalnessMapUv:Rt&&_(S.metalnessMap.channel),roughnessMapUv:b&&_(S.roughnessMap.channel),anisotropyMapUv:dt&&_(S.anisotropyMap.channel),clearcoatMapUv:V&&_(S.clearcoatMap.channel),clearcoatNormalMapUv:at&&_(S.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Z&&_(S.clearcoatRoughnessMap.channel),iridescenceMapUv:it&&_(S.iridescenceMap.channel),iridescenceThicknessMapUv:pt&&_(S.iridescenceThicknessMap.channel),sheenColorMapUv:wt&&_(S.sheenColorMap.channel),sheenRoughnessMapUv:ht&&_(S.sheenRoughnessMap.channel),specularMapUv:Bt&&_(S.specularMap.channel),specularColorMapUv:Ut&&_(S.specularColorMap.channel),specularIntensityMapUv:Zt&&_(S.specularIntensityMap.channel),transmissionMapUv:L&&_(S.transmissionMap.channel),thicknessMapUv:rt&&_(S.thicknessMap.channel),alphaMapUv:$&&_(S.alphaMap.channel),vertexTangents:!!Q.attributes.tangent&&(jt||v),vertexColors:S.vertexColors,vertexAlphas:S.vertexColors===!0&&!!Q.attributes.color&&Q.attributes.color.itemSize===4,pointsUvs:k.isPoints===!0&&!!Q.attributes.uv&&(ie||$),fog:!!q,useFog:S.fog===!0,fogExp2:!!q&&q.isFogExp2,flatShading:S.flatShading===!0,sizeAttenuation:S.sizeAttenuation===!0,logarithmicDepthBuffer:d,reverseDepthBuffer:Ct,skinning:k.isSkinnedMesh===!0,morphTargets:Q.morphAttributes.position!==void 0,morphNormals:Q.morphAttributes.normal!==void 0,morphColors:Q.morphAttributes.color!==void 0,morphTargetsCount:Et,morphTextureStride:Ft,numDirLights:M.directional.length,numPointLights:M.point.length,numSpotLights:M.spot.length,numSpotLightMaps:M.spotLightMap.length,numRectAreaLights:M.rectArea.length,numHemiLights:M.hemi.length,numDirLightShadows:M.directionalShadowMap.length,numPointLightShadows:M.pointShadowMap.length,numSpotLightShadows:M.spotShadowMap.length,numSpotLightShadowsWithMaps:M.numSpotLightShadowsWithMaps,numLightProbes:M.numLightProbes,numClippingPlanes:a.numPlanes,numClipIntersection:a.numIntersection,dithering:S.dithering,shadowMapEnabled:i.shadowMap.enabled&&A.length>0,shadowMapType:i.shadowMap.type,toneMapping:ce,decodeVideoTexture:ie&&S.map.isVideoTexture===!0&&Qt.getTransfer(S.map.colorSpace)===se,decodeVideoTextureEmissive:le&&S.emissiveMap.isVideoTexture===!0&&Qt.getTransfer(S.emissiveMap.colorSpace)===se,premultipliedAlpha:S.premultipliedAlpha,doubleSided:S.side===gn,flipSided:S.side===Xe,useDepthPacking:S.depthPacking>=0,depthPacking:S.depthPacking||0,index0AttributeName:S.index0AttributeName,extensionClipCullDistance:Ot&&S.extensions.clipCullDistance===!0&&n.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Ot&&S.extensions.multiDraw===!0||Vt)&&n.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:n.has("KHR_parallel_shader_compile"),customProgramCacheKey:S.customProgramCacheKey()};return Ae.vertexUv1s=c.has(1),Ae.vertexUv2s=c.has(2),Ae.vertexUv3s=c.has(3),c.clear(),Ae}function f(S){const M=[];if(S.shaderID?M.push(S.shaderID):(M.push(S.customVertexShaderID),M.push(S.customFragmentShaderID)),S.defines!==void 0)for(const A in S.defines)M.push(A),M.push(S.defines[A]);return S.isRawShaderMaterial===!1&&(T(M,S),E(M,S),M.push(i.outputColorSpace)),M.push(S.customProgramCacheKey),M.join()}function T(S,M){S.push(M.precision),S.push(M.outputColorSpace),S.push(M.envMapMode),S.push(M.envMapCubeUVHeight),S.push(M.mapUv),S.push(M.alphaMapUv),S.push(M.lightMapUv),S.push(M.aoMapUv),S.push(M.bumpMapUv),S.push(M.normalMapUv),S.push(M.displacementMapUv),S.push(M.emissiveMapUv),S.push(M.metalnessMapUv),S.push(M.roughnessMapUv),S.push(M.anisotropyMapUv),S.push(M.clearcoatMapUv),S.push(M.clearcoatNormalMapUv),S.push(M.clearcoatRoughnessMapUv),S.push(M.iridescenceMapUv),S.push(M.iridescenceThicknessMapUv),S.push(M.sheenColorMapUv),S.push(M.sheenRoughnessMapUv),S.push(M.specularMapUv),S.push(M.specularColorMapUv),S.push(M.specularIntensityMapUv),S.push(M.transmissionMapUv),S.push(M.thicknessMapUv),S.push(M.combine),S.push(M.fogExp2),S.push(M.sizeAttenuation),S.push(M.morphTargetsCount),S.push(M.morphAttributeCount),S.push(M.numDirLights),S.push(M.numPointLights),S.push(M.numSpotLights),S.push(M.numSpotLightMaps),S.push(M.numHemiLights),S.push(M.numRectAreaLights),S.push(M.numDirLightShadows),S.push(M.numPointLightShadows),S.push(M.numSpotLightShadows),S.push(M.numSpotLightShadowsWithMaps),S.push(M.numLightProbes),S.push(M.shadowMapType),S.push(M.toneMapping),S.push(M.numClippingPlanes),S.push(M.numClipIntersection),S.push(M.depthPacking)}function E(S,M){o.disableAll(),M.supportsVertexTextures&&o.enable(0),M.instancing&&o.enable(1),M.instancingColor&&o.enable(2),M.instancingMorph&&o.enable(3),M.matcap&&o.enable(4),M.envMap&&o.enable(5),M.normalMapObjectSpace&&o.enable(6),M.normalMapTangentSpace&&o.enable(7),M.clearcoat&&o.enable(8),M.iridescence&&o.enable(9),M.alphaTest&&o.enable(10),M.vertexColors&&o.enable(11),M.vertexAlphas&&o.enable(12),M.vertexUv1s&&o.enable(13),M.vertexUv2s&&o.enable(14),M.vertexUv3s&&o.enable(15),M.vertexTangents&&o.enable(16),M.anisotropy&&o.enable(17),M.alphaHash&&o.enable(18),M.batching&&o.enable(19),M.dispersion&&o.enable(20),M.batchingColor&&o.enable(21),S.push(o.mask),o.disableAll(),M.fog&&o.enable(0),M.useFog&&o.enable(1),M.flatShading&&o.enable(2),M.logarithmicDepthBuffer&&o.enable(3),M.reverseDepthBuffer&&o.enable(4),M.skinning&&o.enable(5),M.morphTargets&&o.enable(6),M.morphNormals&&o.enable(7),M.morphColors&&o.enable(8),M.premultipliedAlpha&&o.enable(9),M.shadowMapEnabled&&o.enable(10),M.doubleSided&&o.enable(11),M.flipSided&&o.enable(12),M.useDepthPacking&&o.enable(13),M.dithering&&o.enable(14),M.transmission&&o.enable(15),M.sheen&&o.enable(16),M.opaque&&o.enable(17),M.pointsUvs&&o.enable(18),M.decodeVideoTexture&&o.enable(19),M.decodeVideoTextureEmissive&&o.enable(20),M.alphaToCoverage&&o.enable(21),S.push(o.mask)}function y(S){const M=g[S.type];let A;if(M){const W=mn[M];A=Tr.clone(W.uniforms)}else A=S.uniforms;return A}function D(S,M){let A;for(let W=0,k=h.length;W<k;W++){const q=h[W];if(q.cacheKey===M){A=q,++A.usedTimes;break}}return A===void 0&&(A=new Um(i,M,S,r),h.push(A)),A}function w(S){if(--S.usedTimes===0){const M=h.indexOf(S);h[M]=h[h.length-1],h.pop(),S.destroy()}}function C(S){l.remove(S)}function I(){l.dispose()}return{getParameters:m,getProgramCacheKey:f,getUniforms:y,acquireProgram:D,releaseProgram:w,releaseShaderCache:C,programs:h,dispose:I}}function Bm(){let i=new WeakMap;function t(a){return i.has(a)}function e(a){let o=i.get(a);return o===void 0&&(o={},i.set(a,o)),o}function n(a){i.delete(a)}function s(a,o,l){i.get(a)[o]=l}function r(){i=new WeakMap}return{has:t,get:e,remove:n,update:s,dispose:r}}function zm(i,t){return i.groupOrder!==t.groupOrder?i.groupOrder-t.groupOrder:i.renderOrder!==t.renderOrder?i.renderOrder-t.renderOrder:i.material.id!==t.material.id?i.material.id-t.material.id:i.z!==t.z?i.z-t.z:i.id-t.id}function zl(i,t){return i.groupOrder!==t.groupOrder?i.groupOrder-t.groupOrder:i.renderOrder!==t.renderOrder?i.renderOrder-t.renderOrder:i.z!==t.z?t.z-i.z:i.id-t.id}function kl(){const i=[];let t=0;const e=[],n=[],s=[];function r(){t=0,e.length=0,n.length=0,s.length=0}function a(d,p,u,g,_,m){let f=i[t];return f===void 0?(f={id:d.id,object:d,geometry:p,material:u,groupOrder:g,renderOrder:d.renderOrder,z:_,group:m},i[t]=f):(f.id=d.id,f.object=d,f.geometry=p,f.material=u,f.groupOrder=g,f.renderOrder=d.renderOrder,f.z=_,f.group=m),t++,f}function o(d,p,u,g,_,m){const f=a(d,p,u,g,_,m);u.transmission>0?n.push(f):u.transparent===!0?s.push(f):e.push(f)}function l(d,p,u,g,_,m){const f=a(d,p,u,g,_,m);u.transmission>0?n.unshift(f):u.transparent===!0?s.unshift(f):e.unshift(f)}function c(d,p){e.length>1&&e.sort(d||zm),n.length>1&&n.sort(p||zl),s.length>1&&s.sort(p||zl)}function h(){for(let d=t,p=i.length;d<p;d++){const u=i[d];if(u.id===null)break;u.id=null,u.object=null,u.geometry=null,u.material=null,u.group=null}}return{opaque:e,transmissive:n,transparent:s,init:r,push:o,unshift:l,finish:h,sort:c}}function km(){let i=new WeakMap;function t(n,s){const r=i.get(n);let a;return r===void 0?(a=new kl,i.set(n,[a])):s>=r.length?(a=new kl,r.push(a)):a=r[s],a}function e(){i=new WeakMap}return{get:t,dispose:e}}function Hm(){const i={};return{get:function(t){if(i[t.id]!==void 0)return i[t.id];let e;switch(t.type){case"DirectionalLight":e={direction:new P,color:new ot};break;case"SpotLight":e={position:new P,direction:new P,color:new ot,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":e={position:new P,color:new ot,distance:0,decay:0};break;case"HemisphereLight":e={direction:new P,skyColor:new ot,groundColor:new ot};break;case"RectAreaLight":e={color:new ot,position:new P,halfWidth:new P,halfHeight:new P};break}return i[t.id]=e,e}}}function Vm(){const i={};return{get:function(t){if(i[t.id]!==void 0)return i[t.id];let e;switch(t.type){case"DirectionalLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new St};break;case"SpotLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new St};break;case"PointLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new St,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[t.id]=e,e}}}let Gm=0;function Wm(i,t){return(t.castShadow?2:0)-(i.castShadow?2:0)+(t.map?1:0)-(i.map?1:0)}function Xm(i){const t=new Hm,e=Vm(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)n.probe.push(new P);const s=new P,r=new ne,a=new ne;function o(c){let h=0,d=0,p=0;for(let S=0;S<9;S++)n.probe[S].set(0,0,0);let u=0,g=0,_=0,m=0,f=0,T=0,E=0,y=0,D=0,w=0,C=0;c.sort(Wm);for(let S=0,M=c.length;S<M;S++){const A=c[S],W=A.color,k=A.intensity,q=A.distance,Q=A.shadow&&A.shadow.map?A.shadow.map.texture:null;if(A.isAmbientLight)h+=W.r*k,d+=W.g*k,p+=W.b*k;else if(A.isLightProbe){for(let X=0;X<9;X++)n.probe[X].addScaledVector(A.sh.coefficients[X],k);C++}else if(A.isDirectionalLight){const X=t.get(A);if(X.color.copy(A.color).multiplyScalar(A.intensity),A.castShadow){const tt=A.shadow,H=e.get(A);H.shadowIntensity=tt.intensity,H.shadowBias=tt.bias,H.shadowNormalBias=tt.normalBias,H.shadowRadius=tt.radius,H.shadowMapSize=tt.mapSize,n.directionalShadow[u]=H,n.directionalShadowMap[u]=Q,n.directionalShadowMatrix[u]=A.shadow.matrix,T++}n.directional[u]=X,u++}else if(A.isSpotLight){const X=t.get(A);X.position.setFromMatrixPosition(A.matrixWorld),X.color.copy(W).multiplyScalar(k),X.distance=q,X.coneCos=Math.cos(A.angle),X.penumbraCos=Math.cos(A.angle*(1-A.penumbra)),X.decay=A.decay,n.spot[_]=X;const tt=A.shadow;if(A.map&&(n.spotLightMap[D]=A.map,D++,tt.updateMatrices(A),A.castShadow&&w++),n.spotLightMatrix[_]=tt.matrix,A.castShadow){const H=e.get(A);H.shadowIntensity=tt.intensity,H.shadowBias=tt.bias,H.shadowNormalBias=tt.normalBias,H.shadowRadius=tt.radius,H.shadowMapSize=tt.mapSize,n.spotShadow[_]=H,n.spotShadowMap[_]=Q,y++}_++}else if(A.isRectAreaLight){const X=t.get(A);X.color.copy(W).multiplyScalar(k),X.halfWidth.set(A.width*.5,0,0),X.halfHeight.set(0,A.height*.5,0),n.rectArea[m]=X,m++}else if(A.isPointLight){const X=t.get(A);if(X.color.copy(A.color).multiplyScalar(A.intensity),X.distance=A.distance,X.decay=A.decay,A.castShadow){const tt=A.shadow,H=e.get(A);H.shadowIntensity=tt.intensity,H.shadowBias=tt.bias,H.shadowNormalBias=tt.normalBias,H.shadowRadius=tt.radius,H.shadowMapSize=tt.mapSize,H.shadowCameraNear=tt.camera.near,H.shadowCameraFar=tt.camera.far,n.pointShadow[g]=H,n.pointShadowMap[g]=Q,n.pointShadowMatrix[g]=A.shadow.matrix,E++}n.point[g]=X,g++}else if(A.isHemisphereLight){const X=t.get(A);X.skyColor.copy(A.color).multiplyScalar(k),X.groundColor.copy(A.groundColor).multiplyScalar(k),n.hemi[f]=X,f++}}m>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=ct.LTC_FLOAT_1,n.rectAreaLTC2=ct.LTC_FLOAT_2):(n.rectAreaLTC1=ct.LTC_HALF_1,n.rectAreaLTC2=ct.LTC_HALF_2)),n.ambient[0]=h,n.ambient[1]=d,n.ambient[2]=p;const I=n.hash;(I.directionalLength!==u||I.pointLength!==g||I.spotLength!==_||I.rectAreaLength!==m||I.hemiLength!==f||I.numDirectionalShadows!==T||I.numPointShadows!==E||I.numSpotShadows!==y||I.numSpotMaps!==D||I.numLightProbes!==C)&&(n.directional.length=u,n.spot.length=_,n.rectArea.length=m,n.point.length=g,n.hemi.length=f,n.directionalShadow.length=T,n.directionalShadowMap.length=T,n.pointShadow.length=E,n.pointShadowMap.length=E,n.spotShadow.length=y,n.spotShadowMap.length=y,n.directionalShadowMatrix.length=T,n.pointShadowMatrix.length=E,n.spotLightMatrix.length=y+D-w,n.spotLightMap.length=D,n.numSpotLightShadowsWithMaps=w,n.numLightProbes=C,I.directionalLength=u,I.pointLength=g,I.spotLength=_,I.rectAreaLength=m,I.hemiLength=f,I.numDirectionalShadows=T,I.numPointShadows=E,I.numSpotShadows=y,I.numSpotMaps=D,I.numLightProbes=C,n.version=Gm++)}function l(c,h){let d=0,p=0,u=0,g=0,_=0;const m=h.matrixWorldInverse;for(let f=0,T=c.length;f<T;f++){const E=c[f];if(E.isDirectionalLight){const y=n.directional[d];y.direction.setFromMatrixPosition(E.matrixWorld),s.setFromMatrixPosition(E.target.matrixWorld),y.direction.sub(s),y.direction.transformDirection(m),d++}else if(E.isSpotLight){const y=n.spot[u];y.position.setFromMatrixPosition(E.matrixWorld),y.position.applyMatrix4(m),y.direction.setFromMatrixPosition(E.matrixWorld),s.setFromMatrixPosition(E.target.matrixWorld),y.direction.sub(s),y.direction.transformDirection(m),u++}else if(E.isRectAreaLight){const y=n.rectArea[g];y.position.setFromMatrixPosition(E.matrixWorld),y.position.applyMatrix4(m),a.identity(),r.copy(E.matrixWorld),r.premultiply(m),a.extractRotation(r),y.halfWidth.set(E.width*.5,0,0),y.halfHeight.set(0,E.height*.5,0),y.halfWidth.applyMatrix4(a),y.halfHeight.applyMatrix4(a),g++}else if(E.isPointLight){const y=n.point[p];y.position.setFromMatrixPosition(E.matrixWorld),y.position.applyMatrix4(m),p++}else if(E.isHemisphereLight){const y=n.hemi[_];y.direction.setFromMatrixPosition(E.matrixWorld),y.direction.transformDirection(m),_++}}}return{setup:o,setupView:l,state:n}}function Hl(i){const t=new Xm(i),e=[],n=[];function s(h){c.camera=h,e.length=0,n.length=0}function r(h){e.push(h)}function a(h){n.push(h)}function o(){t.setup(e)}function l(h){t.setupView(e,h)}const c={lightsArray:e,shadowsArray:n,camera:null,lights:t,transmissionRenderTarget:{}};return{init:s,state:c,setupLights:o,setupLightsView:l,pushLight:r,pushShadow:a}}function Ym(i){let t=new WeakMap;function e(s,r=0){const a=t.get(s);let o;return a===void 0?(o=new Hl(i),t.set(s,[o])):r>=a.length?(o=new Hl(i),a.push(o)):o=a[r],o}function n(){t=new WeakMap}return{get:e,dispose:n}}const qm=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,jm=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function Zm(i,t,e){let n=new Ao;const s=new St,r=new St,a=new oe,o=new gu({depthPacking:wh}),l=new _u,c={},h=e.maxTextureSize,d={[Yn]:Xe,[Xe]:Yn,[gn]:gn},p=new He({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new St},radius:{value:4}},vertexShader:qm,fragmentShader:jm}),u=p.clone();u.defines.HORIZONTAL_PASS=1;const g=new ge;g.setAttribute("position",new he(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const _=new be(g,p),m=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Ql;let f=this.type;this.render=function(w,C,I){if(m.enabled===!1||m.autoUpdate===!1&&m.needsUpdate===!1||w.length===0)return;const S=i.getRenderTarget(),M=i.getActiveCubeFace(),A=i.getActiveMipmapLevel(),W=i.state;W.setBlending(Cn),W.buffers.color.setClear(1,1,1,1),W.buffers.depth.setTest(!0),W.setScissorTest(!1);const k=f!==An&&this.type===An,q=f===An&&this.type!==An;for(let Q=0,X=w.length;Q<X;Q++){const tt=w[Q],H=tt.shadow;if(H===void 0){console.warn("THREE.WebGLShadowMap:",tt,"has no shadow.");continue}if(H.autoUpdate===!1&&H.needsUpdate===!1)continue;s.copy(H.mapSize);const st=H.getFrameExtents();if(s.multiply(st),r.copy(H.mapSize),(s.x>h||s.y>h)&&(s.x>h&&(r.x=Math.floor(h/st.x),s.x=r.x*st.x,H.mapSize.x=r.x),s.y>h&&(r.y=Math.floor(h/st.y),s.y=r.y*st.y,H.mapSize.y=r.y)),H.map===null||k===!0||q===!0){const Et=this.type!==An?{minFilter:Je,magFilter:Je}:{};H.map!==null&&H.map.dispose(),H.map=new fn(s.x,s.y,Et),H.map.texture.name=tt.name+".shadowMap",H.camera.updateProjectionMatrix()}i.setRenderTarget(H.map),i.clear();const gt=H.getViewportCount();for(let Et=0;Et<gt;Et++){const Ft=H.getViewport(Et);a.set(r.x*Ft.x,r.y*Ft.y,r.x*Ft.z,r.y*Ft.w),W.viewport(a),H.updateMatrices(tt,Et),n=H.getFrustum(),y(C,I,H.camera,tt,this.type)}H.isPointLightShadow!==!0&&this.type===An&&T(H,I),H.needsUpdate=!1}f=this.type,m.needsUpdate=!1,i.setRenderTarget(S,M,A)};function T(w,C){const I=t.update(_);p.defines.VSM_SAMPLES!==w.blurSamples&&(p.defines.VSM_SAMPLES=w.blurSamples,u.defines.VSM_SAMPLES=w.blurSamples,p.needsUpdate=!0,u.needsUpdate=!0),w.mapPass===null&&(w.mapPass=new fn(s.x,s.y)),p.uniforms.shadow_pass.value=w.map.texture,p.uniforms.resolution.value=w.mapSize,p.uniforms.radius.value=w.radius,i.setRenderTarget(w.mapPass),i.clear(),i.renderBufferDirect(C,null,I,p,_,null),u.uniforms.shadow_pass.value=w.mapPass.texture,u.uniforms.resolution.value=w.mapSize,u.uniforms.radius.value=w.radius,i.setRenderTarget(w.map),i.clear(),i.renderBufferDirect(C,null,I,u,_,null)}function E(w,C,I,S){let M=null;const A=I.isPointLight===!0?w.customDistanceMaterial:w.customDepthMaterial;if(A!==void 0)M=A;else if(M=I.isPointLight===!0?l:o,i.localClippingEnabled&&C.clipShadows===!0&&Array.isArray(C.clippingPlanes)&&C.clippingPlanes.length!==0||C.displacementMap&&C.displacementScale!==0||C.alphaMap&&C.alphaTest>0||C.map&&C.alphaTest>0){const W=M.uuid,k=C.uuid;let q=c[W];q===void 0&&(q={},c[W]=q);let Q=q[k];Q===void 0&&(Q=M.clone(),q[k]=Q,C.addEventListener("dispose",D)),M=Q}if(M.visible=C.visible,M.wireframe=C.wireframe,S===An?M.side=C.shadowSide!==null?C.shadowSide:C.side:M.side=C.shadowSide!==null?C.shadowSide:d[C.side],M.alphaMap=C.alphaMap,M.alphaTest=C.alphaTest,M.map=C.map,M.clipShadows=C.clipShadows,M.clippingPlanes=C.clippingPlanes,M.clipIntersection=C.clipIntersection,M.displacementMap=C.displacementMap,M.displacementScale=C.displacementScale,M.displacementBias=C.displacementBias,M.wireframeLinewidth=C.wireframeLinewidth,M.linewidth=C.linewidth,I.isPointLight===!0&&M.isMeshDistanceMaterial===!0){const W=i.properties.get(M);W.light=I}return M}function y(w,C,I,S,M){if(w.visible===!1)return;if(w.layers.test(C.layers)&&(w.isMesh||w.isLine||w.isPoints)&&(w.castShadow||w.receiveShadow&&M===An)&&(!w.frustumCulled||n.intersectsObject(w))){w.modelViewMatrix.multiplyMatrices(I.matrixWorldInverse,w.matrixWorld);const k=t.update(w),q=w.material;if(Array.isArray(q)){const Q=k.groups;for(let X=0,tt=Q.length;X<tt;X++){const H=Q[X],st=q[H.materialIndex];if(st&&st.visible){const gt=E(w,st,S,M);w.onBeforeShadow(i,w,C,I,k,gt,H),i.renderBufferDirect(I,null,k,gt,w,H),w.onAfterShadow(i,w,C,I,k,gt,H)}}}else if(q.visible){const Q=E(w,q,S,M);w.onBeforeShadow(i,w,C,I,k,Q,null),i.renderBufferDirect(I,null,k,Q,w,null),w.onAfterShadow(i,w,C,I,k,Q,null)}}const W=w.children;for(let k=0,q=W.length;k<q;k++)y(W[k],C,I,S,M)}function D(w){w.target.removeEventListener("dispose",D);for(const I in c){const S=c[I],M=w.target.uuid;M in S&&(S[M].dispose(),delete S[M])}}}const Km={[ba]:Ta,[wa]:Ca,[Aa]:Pa,[qi]:Ra,[Ta]:ba,[Ca]:wa,[Pa]:Aa,[Ra]:qi};function $m(i,t){function e(){let L=!1;const rt=new oe;let G=null;const $=new oe(0,0,0,0);return{setMask:function(ft){G!==ft&&!L&&(i.colorMask(ft,ft,ft,ft),G=ft)},setLocked:function(ft){L=ft},setClear:function(ft,ut,Ot,ce,Ae){Ae===!0&&(ft*=ce,ut*=ce,Ot*=ce),rt.set(ft,ut,Ot,ce),$.equals(rt)===!1&&(i.clearColor(ft,ut,Ot,ce),$.copy(rt))},reset:function(){L=!1,G=null,$.set(-1,0,0,0)}}}function n(){let L=!1,rt=!1,G=null,$=null,ft=null;return{setReversed:function(ut){if(rt!==ut){const Ot=t.get("EXT_clip_control");rt?Ot.clipControlEXT(Ot.LOWER_LEFT_EXT,Ot.ZERO_TO_ONE_EXT):Ot.clipControlEXT(Ot.LOWER_LEFT_EXT,Ot.NEGATIVE_ONE_TO_ONE_EXT);const ce=ft;ft=null,this.setClear(ce)}rt=ut},getReversed:function(){return rt},setTest:function(ut){ut?lt(i.DEPTH_TEST):Ct(i.DEPTH_TEST)},setMask:function(ut){G!==ut&&!L&&(i.depthMask(ut),G=ut)},setFunc:function(ut){if(rt&&(ut=Km[ut]),$!==ut){switch(ut){case ba:i.depthFunc(i.NEVER);break;case Ta:i.depthFunc(i.ALWAYS);break;case wa:i.depthFunc(i.LESS);break;case qi:i.depthFunc(i.LEQUAL);break;case Aa:i.depthFunc(i.EQUAL);break;case Ra:i.depthFunc(i.GEQUAL);break;case Ca:i.depthFunc(i.GREATER);break;case Pa:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}$=ut}},setLocked:function(ut){L=ut},setClear:function(ut){ft!==ut&&(rt&&(ut=1-ut),i.clearDepth(ut),ft=ut)},reset:function(){L=!1,G=null,$=null,ft=null,rt=!1}}}function s(){let L=!1,rt=null,G=null,$=null,ft=null,ut=null,Ot=null,ce=null,Ae=null;return{setTest:function($t){L||($t?lt(i.STENCIL_TEST):Ct(i.STENCIL_TEST))},setMask:function($t){rt!==$t&&!L&&(i.stencilMask($t),rt=$t)},setFunc:function($t,Ye,Qe){(G!==$t||$!==Ye||ft!==Qe)&&(i.stencilFunc($t,Ye,Qe),G=$t,$=Ye,ft=Qe)},setOp:function($t,Ye,Qe){(ut!==$t||Ot!==Ye||ce!==Qe)&&(i.stencilOp($t,Ye,Qe),ut=$t,Ot=Ye,ce=Qe)},setLocked:function($t){L=$t},setClear:function($t){Ae!==$t&&(i.clearStencil($t),Ae=$t)},reset:function(){L=!1,rt=null,G=null,$=null,ft=null,ut=null,Ot=null,ce=null,Ae=null}}}const r=new e,a=new n,o=new s,l=new WeakMap,c=new WeakMap;let h={},d={},p=new WeakMap,u=[],g=null,_=!1,m=null,f=null,T=null,E=null,y=null,D=null,w=null,C=new ot(0,0,0),I=0,S=!1,M=null,A=null,W=null,k=null,q=null;const Q=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let X=!1,tt=0;const H=i.getParameter(i.VERSION);H.indexOf("WebGL")!==-1?(tt=parseFloat(/^WebGL (\d)/.exec(H)[1]),X=tt>=1):H.indexOf("OpenGL ES")!==-1&&(tt=parseFloat(/^OpenGL ES (\d)/.exec(H)[1]),X=tt>=2);let st=null,gt={};const Et=i.getParameter(i.SCISSOR_BOX),Ft=i.getParameter(i.VIEWPORT),Xt=new oe().fromArray(Et),Y=new oe().fromArray(Ft);function nt(L,rt,G,$){const ft=new Uint8Array(4),ut=i.createTexture();i.bindTexture(L,ut),i.texParameteri(L,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(L,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let Ot=0;Ot<G;Ot++)L===i.TEXTURE_3D||L===i.TEXTURE_2D_ARRAY?i.texImage3D(rt,0,i.RGBA,1,1,$,0,i.RGBA,i.UNSIGNED_BYTE,ft):i.texImage2D(rt+Ot,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,ft);return ut}const _t={};_t[i.TEXTURE_2D]=nt(i.TEXTURE_2D,i.TEXTURE_2D,1),_t[i.TEXTURE_CUBE_MAP]=nt(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),_t[i.TEXTURE_2D_ARRAY]=nt(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),_t[i.TEXTURE_3D]=nt(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),a.setClear(1),o.setClear(0),lt(i.DEPTH_TEST),a.setFunc(qi),qt(!1),jt(Lo),lt(i.CULL_FACE),R(Cn);function lt(L){h[L]!==!0&&(i.enable(L),h[L]=!0)}function Ct(L){h[L]!==!1&&(i.disable(L),h[L]=!1)}function Lt(L,rt){return d[L]!==rt?(i.bindFramebuffer(L,rt),d[L]=rt,L===i.DRAW_FRAMEBUFFER&&(d[i.FRAMEBUFFER]=rt),L===i.FRAMEBUFFER&&(d[i.DRAW_FRAMEBUFFER]=rt),!0):!1}function Vt(L,rt){let G=u,$=!1;if(L){G=p.get(rt),G===void 0&&(G=[],p.set(rt,G));const ft=L.textures;if(G.length!==ft.length||G[0]!==i.COLOR_ATTACHMENT0){for(let ut=0,Ot=ft.length;ut<Ot;ut++)G[ut]=i.COLOR_ATTACHMENT0+ut;G.length=ft.length,$=!0}}else G[0]!==i.BACK&&(G[0]=i.BACK,$=!0);$&&i.drawBuffers(G)}function ie(L){return g!==L?(i.useProgram(L),g=L,!0):!1}const Wt={[ii]:i.FUNC_ADD,[Qc]:i.FUNC_SUBTRACT,[th]:i.FUNC_REVERSE_SUBTRACT};Wt[eh]=i.MIN,Wt[nh]=i.MAX;const ue={[ih]:i.ZERO,[sh]:i.ONE,[rh]:i.SRC_COLOR,[ya]:i.SRC_ALPHA,[uh]:i.SRC_ALPHA_SATURATE,[ch]:i.DST_COLOR,[oh]:i.DST_ALPHA,[ah]:i.ONE_MINUS_SRC_COLOR,[Ea]:i.ONE_MINUS_SRC_ALPHA,[hh]:i.ONE_MINUS_DST_COLOR,[lh]:i.ONE_MINUS_DST_ALPHA,[dh]:i.CONSTANT_COLOR,[fh]:i.ONE_MINUS_CONSTANT_COLOR,[ph]:i.CONSTANT_ALPHA,[mh]:i.ONE_MINUS_CONSTANT_ALPHA};function R(L,rt,G,$,ft,ut,Ot,ce,Ae,$t){if(L===Cn){_===!0&&(Ct(i.BLEND),_=!1);return}if(_===!1&&(lt(i.BLEND),_=!0),L!==Jc){if(L!==m||$t!==S){if((f!==ii||y!==ii)&&(i.blendEquation(i.FUNC_ADD),f=ii,y=ii),$t)switch(L){case Vi:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Le:i.blendFunc(i.ONE,i.ONE);break;case Uo:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case Io:i.blendFuncSeparate(i.ZERO,i.SRC_COLOR,i.ZERO,i.SRC_ALPHA);break;default:console.error("THREE.WebGLState: Invalid blending: ",L);break}else switch(L){case Vi:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Le:i.blendFunc(i.SRC_ALPHA,i.ONE);break;case Uo:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case Io:i.blendFunc(i.ZERO,i.SRC_COLOR);break;default:console.error("THREE.WebGLState: Invalid blending: ",L);break}T=null,E=null,D=null,w=null,C.set(0,0,0),I=0,m=L,S=$t}return}ft=ft||rt,ut=ut||G,Ot=Ot||$,(rt!==f||ft!==y)&&(i.blendEquationSeparate(Wt[rt],Wt[ft]),f=rt,y=ft),(G!==T||$!==E||ut!==D||Ot!==w)&&(i.blendFuncSeparate(ue[G],ue[$],ue[ut],ue[Ot]),T=G,E=$,D=ut,w=Ot),(ce.equals(C)===!1||Ae!==I)&&(i.blendColor(ce.r,ce.g,ce.b,Ae),C.copy(ce),I=Ae),m=L,S=!1}function ye(L,rt){L.side===gn?Ct(i.CULL_FACE):lt(i.CULL_FACE);let G=L.side===Xe;rt&&(G=!G),qt(G),L.blending===Vi&&L.transparent===!1?R(Cn):R(L.blending,L.blendEquation,L.blendSrc,L.blendDst,L.blendEquationAlpha,L.blendSrcAlpha,L.blendDstAlpha,L.blendColor,L.blendAlpha,L.premultipliedAlpha),a.setFunc(L.depthFunc),a.setTest(L.depthTest),a.setMask(L.depthWrite),r.setMask(L.colorWrite);const $=L.stencilWrite;o.setTest($),$&&(o.setMask(L.stencilWriteMask),o.setFunc(L.stencilFunc,L.stencilRef,L.stencilFuncMask),o.setOp(L.stencilFail,L.stencilZFail,L.stencilZPass)),le(L.polygonOffset,L.polygonOffsetFactor,L.polygonOffsetUnits),L.alphaToCoverage===!0?lt(i.SAMPLE_ALPHA_TO_COVERAGE):Ct(i.SAMPLE_ALPHA_TO_COVERAGE)}function qt(L){M!==L&&(L?i.frontFace(i.CW):i.frontFace(i.CCW),M=L)}function jt(L){L!==Zc?(lt(i.CULL_FACE),L!==A&&(L===Lo?i.cullFace(i.BACK):L===Kc?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):Ct(i.CULL_FACE),A=L}function At(L){L!==W&&(X&&i.lineWidth(L),W=L)}function le(L,rt,G){L?(lt(i.POLYGON_OFFSET_FILL),(k!==rt||q!==G)&&(i.polygonOffset(rt,G),k=rt,q=G)):Ct(i.POLYGON_OFFSET_FILL)}function Rt(L){L?lt(i.SCISSOR_TEST):Ct(i.SCISSOR_TEST)}function b(L){L===void 0&&(L=i.TEXTURE0+Q-1),st!==L&&(i.activeTexture(L),st=L)}function v(L,rt,G){G===void 0&&(st===null?G=i.TEXTURE0+Q-1:G=st);let $=gt[G];$===void 0&&($={type:void 0,texture:void 0},gt[G]=$),($.type!==L||$.texture!==rt)&&(st!==G&&(i.activeTexture(G),st=G),i.bindTexture(L,rt||_t[L]),$.type=L,$.texture=rt)}function F(){const L=gt[st];L!==void 0&&L.type!==void 0&&(i.bindTexture(L.type,null),L.type=void 0,L.texture=void 0)}function K(){try{i.compressedTexImage2D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function J(){try{i.compressedTexImage3D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function j(){try{i.texSubImage2D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function Tt(){try{i.texSubImage3D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function dt(){try{i.compressedTexSubImage2D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function V(){try{i.compressedTexSubImage3D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function at(){try{i.texStorage2D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function Z(){try{i.texStorage3D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function it(){try{i.texImage2D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function pt(){try{i.texImage3D.apply(i,arguments)}catch(L){console.error("THREE.WebGLState:",L)}}function wt(L){Xt.equals(L)===!1&&(i.scissor(L.x,L.y,L.z,L.w),Xt.copy(L))}function ht(L){Y.equals(L)===!1&&(i.viewport(L.x,L.y,L.z,L.w),Y.copy(L))}function Bt(L,rt){let G=c.get(rt);G===void 0&&(G=new WeakMap,c.set(rt,G));let $=G.get(L);$===void 0&&($=i.getUniformBlockIndex(rt,L.name),G.set(L,$))}function Ut(L,rt){const $=c.get(rt).get(L);l.get(rt)!==$&&(i.uniformBlockBinding(rt,$,L.__bindingPointIndex),l.set(rt,$))}function Zt(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),a.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),h={},st=null,gt={},d={},p=new WeakMap,u=[],g=null,_=!1,m=null,f=null,T=null,E=null,y=null,D=null,w=null,C=new ot(0,0,0),I=0,S=!1,M=null,A=null,W=null,k=null,q=null,Xt.set(0,0,i.canvas.width,i.canvas.height),Y.set(0,0,i.canvas.width,i.canvas.height),r.reset(),a.reset(),o.reset()}return{buffers:{color:r,depth:a,stencil:o},enable:lt,disable:Ct,bindFramebuffer:Lt,drawBuffers:Vt,useProgram:ie,setBlending:R,setMaterial:ye,setFlipSided:qt,setCullFace:jt,setLineWidth:At,setPolygonOffset:le,setScissorTest:Rt,activeTexture:b,bindTexture:v,unbindTexture:F,compressedTexImage2D:K,compressedTexImage3D:J,texImage2D:it,texImage3D:pt,updateUBOMapping:Bt,uniformBlockBinding:Ut,texStorage2D:at,texStorage3D:Z,texSubImage2D:j,texSubImage3D:Tt,compressedTexSubImage2D:dt,compressedTexSubImage3D:V,scissor:wt,viewport:ht,reset:Zt}}function Jm(i,t,e,n,s,r,a){const o=t.has("WEBGL_multisampled_render_to_texture")?t.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new St,h=new WeakMap;let d;const p=new WeakMap;let u=!1;try{u=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function g(b,v){return u?new OffscreenCanvas(b,v):br("canvas")}function _(b,v,F){let K=1;const J=Rt(b);if((J.width>F||J.height>F)&&(K=F/Math.max(J.width,J.height)),K<1)if(typeof HTMLImageElement<"u"&&b instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&b instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&b instanceof ImageBitmap||typeof VideoFrame<"u"&&b instanceof VideoFrame){const j=Math.floor(K*J.width),Tt=Math.floor(K*J.height);d===void 0&&(d=g(j,Tt));const dt=v?g(j,Tt):d;return dt.width=j,dt.height=Tt,dt.getContext("2d").drawImage(b,0,0,j,Tt),console.warn("THREE.WebGLRenderer: Texture has been resized from ("+J.width+"x"+J.height+") to ("+j+"x"+Tt+")."),dt}else return"data"in b&&console.warn("THREE.WebGLRenderer: Image in DataTexture is too big ("+J.width+"x"+J.height+")."),b;return b}function m(b){return b.generateMipmaps}function f(b){i.generateMipmap(b)}function T(b){return b.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:b.isWebGL3DRenderTarget?i.TEXTURE_3D:b.isWebGLArrayRenderTarget||b.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function E(b,v,F,K,J=!1){if(b!==null){if(i[b]!==void 0)return i[b];console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '"+b+"'")}let j=v;if(v===i.RED&&(F===i.FLOAT&&(j=i.R32F),F===i.HALF_FLOAT&&(j=i.R16F),F===i.UNSIGNED_BYTE&&(j=i.R8)),v===i.RED_INTEGER&&(F===i.UNSIGNED_BYTE&&(j=i.R8UI),F===i.UNSIGNED_SHORT&&(j=i.R16UI),F===i.UNSIGNED_INT&&(j=i.R32UI),F===i.BYTE&&(j=i.R8I),F===i.SHORT&&(j=i.R16I),F===i.INT&&(j=i.R32I)),v===i.RG&&(F===i.FLOAT&&(j=i.RG32F),F===i.HALF_FLOAT&&(j=i.RG16F),F===i.UNSIGNED_BYTE&&(j=i.RG8)),v===i.RG_INTEGER&&(F===i.UNSIGNED_BYTE&&(j=i.RG8UI),F===i.UNSIGNED_SHORT&&(j=i.RG16UI),F===i.UNSIGNED_INT&&(j=i.RG32UI),F===i.BYTE&&(j=i.RG8I),F===i.SHORT&&(j=i.RG16I),F===i.INT&&(j=i.RG32I)),v===i.RGB_INTEGER&&(F===i.UNSIGNED_BYTE&&(j=i.RGB8UI),F===i.UNSIGNED_SHORT&&(j=i.RGB16UI),F===i.UNSIGNED_INT&&(j=i.RGB32UI),F===i.BYTE&&(j=i.RGB8I),F===i.SHORT&&(j=i.RGB16I),F===i.INT&&(j=i.RGB32I)),v===i.RGBA_INTEGER&&(F===i.UNSIGNED_BYTE&&(j=i.RGBA8UI),F===i.UNSIGNED_SHORT&&(j=i.RGBA16UI),F===i.UNSIGNED_INT&&(j=i.RGBA32UI),F===i.BYTE&&(j=i.RGBA8I),F===i.SHORT&&(j=i.RGBA16I),F===i.INT&&(j=i.RGBA32I)),v===i.RGB&&F===i.UNSIGNED_INT_5_9_9_9_REV&&(j=i.RGB9_E5),v===i.RGBA){const Tt=J?yr:Qt.getTransfer(K);F===i.FLOAT&&(j=i.RGBA32F),F===i.HALF_FLOAT&&(j=i.RGBA16F),F===i.UNSIGNED_BYTE&&(j=Tt===se?i.SRGB8_ALPHA8:i.RGBA8),F===i.UNSIGNED_SHORT_4_4_4_4&&(j=i.RGBA4),F===i.UNSIGNED_SHORT_5_5_5_1&&(j=i.RGB5_A1)}return(j===i.R16F||j===i.R32F||j===i.RG16F||j===i.RG32F||j===i.RGBA16F||j===i.RGBA32F)&&t.get("EXT_color_buffer_float"),j}function y(b,v){let F;return b?v===null||v===li||v===Ki?F=i.DEPTH24_STENCIL8:v===xn?F=i.DEPTH32F_STENCIL8:v===Ms&&(F=i.DEPTH24_STENCIL8,console.warn("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):v===null||v===li||v===Ki?F=i.DEPTH_COMPONENT24:v===xn?F=i.DEPTH_COMPONENT32F:v===Ms&&(F=i.DEPTH_COMPONENT16),F}function D(b,v){return m(b)===!0||b.isFramebufferTexture&&b.minFilter!==Je&&b.minFilter!==vn?Math.log2(Math.max(v.width,v.height))+1:b.mipmaps!==void 0&&b.mipmaps.length>0?b.mipmaps.length:b.isCompressedTexture&&Array.isArray(b.image)?v.mipmaps.length:1}function w(b){const v=b.target;v.removeEventListener("dispose",w),I(v),v.isVideoTexture&&h.delete(v)}function C(b){const v=b.target;v.removeEventListener("dispose",C),M(v)}function I(b){const v=n.get(b);if(v.__webglInit===void 0)return;const F=b.source,K=p.get(F);if(K){const J=K[v.__cacheKey];J.usedTimes--,J.usedTimes===0&&S(b),Object.keys(K).length===0&&p.delete(F)}n.remove(b)}function S(b){const v=n.get(b);i.deleteTexture(v.__webglTexture);const F=b.source,K=p.get(F);delete K[v.__cacheKey],a.memory.textures--}function M(b){const v=n.get(b);if(b.depthTexture&&(b.depthTexture.dispose(),n.remove(b.depthTexture)),b.isWebGLCubeRenderTarget)for(let K=0;K<6;K++){if(Array.isArray(v.__webglFramebuffer[K]))for(let J=0;J<v.__webglFramebuffer[K].length;J++)i.deleteFramebuffer(v.__webglFramebuffer[K][J]);else i.deleteFramebuffer(v.__webglFramebuffer[K]);v.__webglDepthbuffer&&i.deleteRenderbuffer(v.__webglDepthbuffer[K])}else{if(Array.isArray(v.__webglFramebuffer))for(let K=0;K<v.__webglFramebuffer.length;K++)i.deleteFramebuffer(v.__webglFramebuffer[K]);else i.deleteFramebuffer(v.__webglFramebuffer);if(v.__webglDepthbuffer&&i.deleteRenderbuffer(v.__webglDepthbuffer),v.__webglMultisampledFramebuffer&&i.deleteFramebuffer(v.__webglMultisampledFramebuffer),v.__webglColorRenderbuffer)for(let K=0;K<v.__webglColorRenderbuffer.length;K++)v.__webglColorRenderbuffer[K]&&i.deleteRenderbuffer(v.__webglColorRenderbuffer[K]);v.__webglDepthRenderbuffer&&i.deleteRenderbuffer(v.__webglDepthRenderbuffer)}const F=b.textures;for(let K=0,J=F.length;K<J;K++){const j=n.get(F[K]);j.__webglTexture&&(i.deleteTexture(j.__webglTexture),a.memory.textures--),n.remove(F[K])}n.remove(b)}let A=0;function W(){A=0}function k(){const b=A;return b>=s.maxTextures&&console.warn("THREE.WebGLTextures: Trying to use "+b+" texture units while this GPU supports only "+s.maxTextures),A+=1,b}function q(b){const v=[];return v.push(b.wrapS),v.push(b.wrapT),v.push(b.wrapR||0),v.push(b.magFilter),v.push(b.minFilter),v.push(b.anisotropy),v.push(b.internalFormat),v.push(b.format),v.push(b.type),v.push(b.generateMipmaps),v.push(b.premultiplyAlpha),v.push(b.flipY),v.push(b.unpackAlignment),v.push(b.colorSpace),v.join()}function Q(b,v){const F=n.get(b);if(b.isVideoTexture&&At(b),b.isRenderTargetTexture===!1&&b.version>0&&F.__version!==b.version){const K=b.image;if(K===null)console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");else if(K.complete===!1)console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");else{Y(F,b,v);return}}e.bindTexture(i.TEXTURE_2D,F.__webglTexture,i.TEXTURE0+v)}function X(b,v){const F=n.get(b);if(b.version>0&&F.__version!==b.version){Y(F,b,v);return}e.bindTexture(i.TEXTURE_2D_ARRAY,F.__webglTexture,i.TEXTURE0+v)}function tt(b,v){const F=n.get(b);if(b.version>0&&F.__version!==b.version){Y(F,b,v);return}e.bindTexture(i.TEXTURE_3D,F.__webglTexture,i.TEXTURE0+v)}function H(b,v){const F=n.get(b);if(b.version>0&&F.__version!==b.version){nt(F,b,v);return}e.bindTexture(i.TEXTURE_CUBE_MAP,F.__webglTexture,i.TEXTURE0+v)}const st={[Ua]:i.REPEAT,[ri]:i.CLAMP_TO_EDGE,[Ia]:i.MIRRORED_REPEAT},gt={[Je]:i.NEAREST,[bh]:i.NEAREST_MIPMAP_NEAREST,[Is]:i.NEAREST_MIPMAP_LINEAR,[vn]:i.LINEAR,[Br]:i.LINEAR_MIPMAP_NEAREST,[ai]:i.LINEAR_MIPMAP_LINEAR},Et={[Rh]:i.NEVER,[Ih]:i.ALWAYS,[Ch]:i.LESS,[fc]:i.LEQUAL,[Ph]:i.EQUAL,[Uh]:i.GEQUAL,[Dh]:i.GREATER,[Lh]:i.NOTEQUAL};function Ft(b,v){if(v.type===xn&&t.has("OES_texture_float_linear")===!1&&(v.magFilter===vn||v.magFilter===Br||v.magFilter===Is||v.magFilter===ai||v.minFilter===vn||v.minFilter===Br||v.minFilter===Is||v.minFilter===ai)&&console.warn("THREE.WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(b,i.TEXTURE_WRAP_S,st[v.wrapS]),i.texParameteri(b,i.TEXTURE_WRAP_T,st[v.wrapT]),(b===i.TEXTURE_3D||b===i.TEXTURE_2D_ARRAY)&&i.texParameteri(b,i.TEXTURE_WRAP_R,st[v.wrapR]),i.texParameteri(b,i.TEXTURE_MAG_FILTER,gt[v.magFilter]),i.texParameteri(b,i.TEXTURE_MIN_FILTER,gt[v.minFilter]),v.compareFunction&&(i.texParameteri(b,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(b,i.TEXTURE_COMPARE_FUNC,Et[v.compareFunction])),t.has("EXT_texture_filter_anisotropic")===!0){if(v.magFilter===Je||v.minFilter!==Is&&v.minFilter!==ai||v.type===xn&&t.has("OES_texture_float_linear")===!1)return;if(v.anisotropy>1||n.get(v).__currentAnisotropy){const F=t.get("EXT_texture_filter_anisotropic");i.texParameterf(b,F.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(v.anisotropy,s.getMaxAnisotropy())),n.get(v).__currentAnisotropy=v.anisotropy}}}function Xt(b,v){let F=!1;b.__webglInit===void 0&&(b.__webglInit=!0,v.addEventListener("dispose",w));const K=v.source;let J=p.get(K);J===void 0&&(J={},p.set(K,J));const j=q(v);if(j!==b.__cacheKey){J[j]===void 0&&(J[j]={texture:i.createTexture(),usedTimes:0},a.memory.textures++,F=!0),J[j].usedTimes++;const Tt=J[b.__cacheKey];Tt!==void 0&&(J[b.__cacheKey].usedTimes--,Tt.usedTimes===0&&S(v)),b.__cacheKey=j,b.__webglTexture=J[j].texture}return F}function Y(b,v,F){let K=i.TEXTURE_2D;(v.isDataArrayTexture||v.isCompressedArrayTexture)&&(K=i.TEXTURE_2D_ARRAY),v.isData3DTexture&&(K=i.TEXTURE_3D);const J=Xt(b,v),j=v.source;e.bindTexture(K,b.__webglTexture,i.TEXTURE0+F);const Tt=n.get(j);if(j.version!==Tt.__version||J===!0){e.activeTexture(i.TEXTURE0+F);const dt=Qt.getPrimaries(Qt.workingColorSpace),V=v.colorSpace===Vn?null:Qt.getPrimaries(v.colorSpace),at=v.colorSpace===Vn||dt===V?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,v.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,v.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,at);let Z=_(v.image,!1,s.maxTextureSize);Z=le(v,Z);const it=r.convert(v.format,v.colorSpace),pt=r.convert(v.type);let wt=E(v.internalFormat,it,pt,v.colorSpace,v.isVideoTexture);Ft(K,v);let ht;const Bt=v.mipmaps,Ut=v.isVideoTexture!==!0,Zt=Tt.__version===void 0||J===!0,L=j.dataReady,rt=D(v,Z);if(v.isDepthTexture)wt=y(v.format===$i,v.type),Zt&&(Ut?e.texStorage2D(i.TEXTURE_2D,1,wt,Z.width,Z.height):e.texImage2D(i.TEXTURE_2D,0,wt,Z.width,Z.height,0,it,pt,null));else if(v.isDataTexture)if(Bt.length>0){Ut&&Zt&&e.texStorage2D(i.TEXTURE_2D,rt,wt,Bt[0].width,Bt[0].height);for(let G=0,$=Bt.length;G<$;G++)ht=Bt[G],Ut?L&&e.texSubImage2D(i.TEXTURE_2D,G,0,0,ht.width,ht.height,it,pt,ht.data):e.texImage2D(i.TEXTURE_2D,G,wt,ht.width,ht.height,0,it,pt,ht.data);v.generateMipmaps=!1}else Ut?(Zt&&e.texStorage2D(i.TEXTURE_2D,rt,wt,Z.width,Z.height),L&&e.texSubImage2D(i.TEXTURE_2D,0,0,0,Z.width,Z.height,it,pt,Z.data)):e.texImage2D(i.TEXTURE_2D,0,wt,Z.width,Z.height,0,it,pt,Z.data);else if(v.isCompressedTexture)if(v.isCompressedArrayTexture){Ut&&Zt&&e.texStorage3D(i.TEXTURE_2D_ARRAY,rt,wt,Bt[0].width,Bt[0].height,Z.depth);for(let G=0,$=Bt.length;G<$;G++)if(ht=Bt[G],v.format!==dn)if(it!==null)if(Ut){if(L)if(v.layerUpdates.size>0){const ft=_l(ht.width,ht.height,v.format,v.type);for(const ut of v.layerUpdates){const Ot=ht.data.subarray(ut*ft/ht.data.BYTES_PER_ELEMENT,(ut+1)*ft/ht.data.BYTES_PER_ELEMENT);e.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,G,0,0,ut,ht.width,ht.height,1,it,Ot)}v.clearLayerUpdates()}else e.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,G,0,0,0,ht.width,ht.height,Z.depth,it,ht.data)}else e.compressedTexImage3D(i.TEXTURE_2D_ARRAY,G,wt,ht.width,ht.height,Z.depth,0,ht.data,0,0);else console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else Ut?L&&e.texSubImage3D(i.TEXTURE_2D_ARRAY,G,0,0,0,ht.width,ht.height,Z.depth,it,pt,ht.data):e.texImage3D(i.TEXTURE_2D_ARRAY,G,wt,ht.width,ht.height,Z.depth,0,it,pt,ht.data)}else{Ut&&Zt&&e.texStorage2D(i.TEXTURE_2D,rt,wt,Bt[0].width,Bt[0].height);for(let G=0,$=Bt.length;G<$;G++)ht=Bt[G],v.format!==dn?it!==null?Ut?L&&e.compressedTexSubImage2D(i.TEXTURE_2D,G,0,0,ht.width,ht.height,it,ht.data):e.compressedTexImage2D(i.TEXTURE_2D,G,wt,ht.width,ht.height,0,ht.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):Ut?L&&e.texSubImage2D(i.TEXTURE_2D,G,0,0,ht.width,ht.height,it,pt,ht.data):e.texImage2D(i.TEXTURE_2D,G,wt,ht.width,ht.height,0,it,pt,ht.data)}else if(v.isDataArrayTexture)if(Ut){if(Zt&&e.texStorage3D(i.TEXTURE_2D_ARRAY,rt,wt,Z.width,Z.height,Z.depth),L)if(v.layerUpdates.size>0){const G=_l(Z.width,Z.height,v.format,v.type);for(const $ of v.layerUpdates){const ft=Z.data.subarray($*G/Z.data.BYTES_PER_ELEMENT,($+1)*G/Z.data.BYTES_PER_ELEMENT);e.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,$,Z.width,Z.height,1,it,pt,ft)}v.clearLayerUpdates()}else e.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,Z.width,Z.height,Z.depth,it,pt,Z.data)}else e.texImage3D(i.TEXTURE_2D_ARRAY,0,wt,Z.width,Z.height,Z.depth,0,it,pt,Z.data);else if(v.isData3DTexture)Ut?(Zt&&e.texStorage3D(i.TEXTURE_3D,rt,wt,Z.width,Z.height,Z.depth),L&&e.texSubImage3D(i.TEXTURE_3D,0,0,0,0,Z.width,Z.height,Z.depth,it,pt,Z.data)):e.texImage3D(i.TEXTURE_3D,0,wt,Z.width,Z.height,Z.depth,0,it,pt,Z.data);else if(v.isFramebufferTexture){if(Zt)if(Ut)e.texStorage2D(i.TEXTURE_2D,rt,wt,Z.width,Z.height);else{let G=Z.width,$=Z.height;for(let ft=0;ft<rt;ft++)e.texImage2D(i.TEXTURE_2D,ft,wt,G,$,0,it,pt,null),G>>=1,$>>=1}}else if(Bt.length>0){if(Ut&&Zt){const G=Rt(Bt[0]);e.texStorage2D(i.TEXTURE_2D,rt,wt,G.width,G.height)}for(let G=0,$=Bt.length;G<$;G++)ht=Bt[G],Ut?L&&e.texSubImage2D(i.TEXTURE_2D,G,0,0,it,pt,ht):e.texImage2D(i.TEXTURE_2D,G,wt,it,pt,ht);v.generateMipmaps=!1}else if(Ut){if(Zt){const G=Rt(Z);e.texStorage2D(i.TEXTURE_2D,rt,wt,G.width,G.height)}L&&e.texSubImage2D(i.TEXTURE_2D,0,0,0,it,pt,Z)}else e.texImage2D(i.TEXTURE_2D,0,wt,it,pt,Z);m(v)&&f(K),Tt.__version=j.version,v.onUpdate&&v.onUpdate(v)}b.__version=v.version}function nt(b,v,F){if(v.image.length!==6)return;const K=Xt(b,v),J=v.source;e.bindTexture(i.TEXTURE_CUBE_MAP,b.__webglTexture,i.TEXTURE0+F);const j=n.get(J);if(J.version!==j.__version||K===!0){e.activeTexture(i.TEXTURE0+F);const Tt=Qt.getPrimaries(Qt.workingColorSpace),dt=v.colorSpace===Vn?null:Qt.getPrimaries(v.colorSpace),V=v.colorSpace===Vn||Tt===dt?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,v.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,v.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,V);const at=v.isCompressedTexture||v.image[0].isCompressedTexture,Z=v.image[0]&&v.image[0].isDataTexture,it=[];for(let $=0;$<6;$++)!at&&!Z?it[$]=_(v.image[$],!0,s.maxCubemapSize):it[$]=Z?v.image[$].image:v.image[$],it[$]=le(v,it[$]);const pt=it[0],wt=r.convert(v.format,v.colorSpace),ht=r.convert(v.type),Bt=E(v.internalFormat,wt,ht,v.colorSpace),Ut=v.isVideoTexture!==!0,Zt=j.__version===void 0||K===!0,L=J.dataReady;let rt=D(v,pt);Ft(i.TEXTURE_CUBE_MAP,v);let G;if(at){Ut&&Zt&&e.texStorage2D(i.TEXTURE_CUBE_MAP,rt,Bt,pt.width,pt.height);for(let $=0;$<6;$++){G=it[$].mipmaps;for(let ft=0;ft<G.length;ft++){const ut=G[ft];v.format!==dn?wt!==null?Ut?L&&e.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,ft,0,0,ut.width,ut.height,wt,ut.data):e.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,ft,Bt,ut.width,ut.height,0,ut.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):Ut?L&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,ft,0,0,ut.width,ut.height,wt,ht,ut.data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,ft,Bt,ut.width,ut.height,0,wt,ht,ut.data)}}}else{if(G=v.mipmaps,Ut&&Zt){G.length>0&&rt++;const $=Rt(it[0]);e.texStorage2D(i.TEXTURE_CUBE_MAP,rt,Bt,$.width,$.height)}for(let $=0;$<6;$++)if(Z){Ut?L&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,0,0,0,it[$].width,it[$].height,wt,ht,it[$].data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,0,Bt,it[$].width,it[$].height,0,wt,ht,it[$].data);for(let ft=0;ft<G.length;ft++){const Ot=G[ft].image[$].image;Ut?L&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,ft+1,0,0,Ot.width,Ot.height,wt,ht,Ot.data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,ft+1,Bt,Ot.width,Ot.height,0,wt,ht,Ot.data)}}else{Ut?L&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,0,0,0,wt,ht,it[$]):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,0,Bt,wt,ht,it[$]);for(let ft=0;ft<G.length;ft++){const ut=G[ft];Ut?L&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,ft+1,0,0,wt,ht,ut.image[$]):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+$,ft+1,Bt,wt,ht,ut.image[$])}}}m(v)&&f(i.TEXTURE_CUBE_MAP),j.__version=J.version,v.onUpdate&&v.onUpdate(v)}b.__version=v.version}function _t(b,v,F,K,J,j){const Tt=r.convert(F.format,F.colorSpace),dt=r.convert(F.type),V=E(F.internalFormat,Tt,dt,F.colorSpace),at=n.get(v),Z=n.get(F);if(Z.__renderTarget=v,!at.__hasExternalTextures){const it=Math.max(1,v.width>>j),pt=Math.max(1,v.height>>j);J===i.TEXTURE_3D||J===i.TEXTURE_2D_ARRAY?e.texImage3D(J,j,V,it,pt,v.depth,0,Tt,dt,null):e.texImage2D(J,j,V,it,pt,0,Tt,dt,null)}e.bindFramebuffer(i.FRAMEBUFFER,b),jt(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,K,J,Z.__webglTexture,0,qt(v)):(J===i.TEXTURE_2D||J>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&J<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,K,J,Z.__webglTexture,j),e.bindFramebuffer(i.FRAMEBUFFER,null)}function lt(b,v,F){if(i.bindRenderbuffer(i.RENDERBUFFER,b),v.depthBuffer){const K=v.depthTexture,J=K&&K.isDepthTexture?K.type:null,j=y(v.stencilBuffer,J),Tt=v.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,dt=qt(v);jt(v)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,dt,j,v.width,v.height):F?i.renderbufferStorageMultisample(i.RENDERBUFFER,dt,j,v.width,v.height):i.renderbufferStorage(i.RENDERBUFFER,j,v.width,v.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Tt,i.RENDERBUFFER,b)}else{const K=v.textures;for(let J=0;J<K.length;J++){const j=K[J],Tt=r.convert(j.format,j.colorSpace),dt=r.convert(j.type),V=E(j.internalFormat,Tt,dt,j.colorSpace),at=qt(v);F&&jt(v)===!1?i.renderbufferStorageMultisample(i.RENDERBUFFER,at,V,v.width,v.height):jt(v)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,at,V,v.width,v.height):i.renderbufferStorage(i.RENDERBUFFER,V,v.width,v.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function Ct(b,v){if(v&&v.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(e.bindFramebuffer(i.FRAMEBUFFER,b),!(v.depthTexture&&v.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const K=n.get(v.depthTexture);K.__renderTarget=v,(!K.__webglTexture||v.depthTexture.image.width!==v.width||v.depthTexture.image.height!==v.height)&&(v.depthTexture.image.width=v.width,v.depthTexture.image.height=v.height,v.depthTexture.needsUpdate=!0),Q(v.depthTexture,0);const J=K.__webglTexture,j=qt(v);if(v.depthTexture.format===Gi)jt(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,J,0,j):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_ATTACHMENT,i.TEXTURE_2D,J,0);else if(v.depthTexture.format===$i)jt(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,J,0,j):i.framebufferTexture2D(i.FRAMEBUFFER,i.DEPTH_STENCIL_ATTACHMENT,i.TEXTURE_2D,J,0);else throw new Error("Unknown depthTexture format")}function Lt(b){const v=n.get(b),F=b.isWebGLCubeRenderTarget===!0;if(v.__boundDepthTexture!==b.depthTexture){const K=b.depthTexture;if(v.__depthDisposeCallback&&v.__depthDisposeCallback(),K){const J=()=>{delete v.__boundDepthTexture,delete v.__depthDisposeCallback,K.removeEventListener("dispose",J)};K.addEventListener("dispose",J),v.__depthDisposeCallback=J}v.__boundDepthTexture=K}if(b.depthTexture&&!v.__autoAllocateDepthBuffer){if(F)throw new Error("target.depthTexture not supported in Cube render targets");Ct(v.__webglFramebuffer,b)}else if(F){v.__webglDepthbuffer=[];for(let K=0;K<6;K++)if(e.bindFramebuffer(i.FRAMEBUFFER,v.__webglFramebuffer[K]),v.__webglDepthbuffer[K]===void 0)v.__webglDepthbuffer[K]=i.createRenderbuffer(),lt(v.__webglDepthbuffer[K],b,!1);else{const J=b.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,j=v.__webglDepthbuffer[K];i.bindRenderbuffer(i.RENDERBUFFER,j),i.framebufferRenderbuffer(i.FRAMEBUFFER,J,i.RENDERBUFFER,j)}}else if(e.bindFramebuffer(i.FRAMEBUFFER,v.__webglFramebuffer),v.__webglDepthbuffer===void 0)v.__webglDepthbuffer=i.createRenderbuffer(),lt(v.__webglDepthbuffer,b,!1);else{const K=b.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,J=v.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,J),i.framebufferRenderbuffer(i.FRAMEBUFFER,K,i.RENDERBUFFER,J)}e.bindFramebuffer(i.FRAMEBUFFER,null)}function Vt(b,v,F){const K=n.get(b);v!==void 0&&_t(K.__webglFramebuffer,b,b.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),F!==void 0&&Lt(b)}function ie(b){const v=b.texture,F=n.get(b),K=n.get(v);b.addEventListener("dispose",C);const J=b.textures,j=b.isWebGLCubeRenderTarget===!0,Tt=J.length>1;if(Tt||(K.__webglTexture===void 0&&(K.__webglTexture=i.createTexture()),K.__version=v.version,a.memory.textures++),j){F.__webglFramebuffer=[];for(let dt=0;dt<6;dt++)if(v.mipmaps&&v.mipmaps.length>0){F.__webglFramebuffer[dt]=[];for(let V=0;V<v.mipmaps.length;V++)F.__webglFramebuffer[dt][V]=i.createFramebuffer()}else F.__webglFramebuffer[dt]=i.createFramebuffer()}else{if(v.mipmaps&&v.mipmaps.length>0){F.__webglFramebuffer=[];for(let dt=0;dt<v.mipmaps.length;dt++)F.__webglFramebuffer[dt]=i.createFramebuffer()}else F.__webglFramebuffer=i.createFramebuffer();if(Tt)for(let dt=0,V=J.length;dt<V;dt++){const at=n.get(J[dt]);at.__webglTexture===void 0&&(at.__webglTexture=i.createTexture(),a.memory.textures++)}if(b.samples>0&&jt(b)===!1){F.__webglMultisampledFramebuffer=i.createFramebuffer(),F.__webglColorRenderbuffer=[],e.bindFramebuffer(i.FRAMEBUFFER,F.__webglMultisampledFramebuffer);for(let dt=0;dt<J.length;dt++){const V=J[dt];F.__webglColorRenderbuffer[dt]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,F.__webglColorRenderbuffer[dt]);const at=r.convert(V.format,V.colorSpace),Z=r.convert(V.type),it=E(V.internalFormat,at,Z,V.colorSpace,b.isXRRenderTarget===!0),pt=qt(b);i.renderbufferStorageMultisample(i.RENDERBUFFER,pt,it,b.width,b.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+dt,i.RENDERBUFFER,F.__webglColorRenderbuffer[dt])}i.bindRenderbuffer(i.RENDERBUFFER,null),b.depthBuffer&&(F.__webglDepthRenderbuffer=i.createRenderbuffer(),lt(F.__webglDepthRenderbuffer,b,!0)),e.bindFramebuffer(i.FRAMEBUFFER,null)}}if(j){e.bindTexture(i.TEXTURE_CUBE_MAP,K.__webglTexture),Ft(i.TEXTURE_CUBE_MAP,v);for(let dt=0;dt<6;dt++)if(v.mipmaps&&v.mipmaps.length>0)for(let V=0;V<v.mipmaps.length;V++)_t(F.__webglFramebuffer[dt][V],b,v,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+dt,V);else _t(F.__webglFramebuffer[dt],b,v,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+dt,0);m(v)&&f(i.TEXTURE_CUBE_MAP),e.unbindTexture()}else if(Tt){for(let dt=0,V=J.length;dt<V;dt++){const at=J[dt],Z=n.get(at);e.bindTexture(i.TEXTURE_2D,Z.__webglTexture),Ft(i.TEXTURE_2D,at),_t(F.__webglFramebuffer,b,at,i.COLOR_ATTACHMENT0+dt,i.TEXTURE_2D,0),m(at)&&f(i.TEXTURE_2D)}e.unbindTexture()}else{let dt=i.TEXTURE_2D;if((b.isWebGL3DRenderTarget||b.isWebGLArrayRenderTarget)&&(dt=b.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),e.bindTexture(dt,K.__webglTexture),Ft(dt,v),v.mipmaps&&v.mipmaps.length>0)for(let V=0;V<v.mipmaps.length;V++)_t(F.__webglFramebuffer[V],b,v,i.COLOR_ATTACHMENT0,dt,V);else _t(F.__webglFramebuffer,b,v,i.COLOR_ATTACHMENT0,dt,0);m(v)&&f(dt),e.unbindTexture()}b.depthBuffer&&Lt(b)}function Wt(b){const v=b.textures;for(let F=0,K=v.length;F<K;F++){const J=v[F];if(m(J)){const j=T(b),Tt=n.get(J).__webglTexture;e.bindTexture(j,Tt),f(j),e.unbindTexture()}}}const ue=[],R=[];function ye(b){if(b.samples>0){if(jt(b)===!1){const v=b.textures,F=b.width,K=b.height;let J=i.COLOR_BUFFER_BIT;const j=b.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Tt=n.get(b),dt=v.length>1;if(dt)for(let V=0;V<v.length;V++)e.bindFramebuffer(i.FRAMEBUFFER,Tt.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+V,i.RENDERBUFFER,null),e.bindFramebuffer(i.FRAMEBUFFER,Tt.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+V,i.TEXTURE_2D,null,0);e.bindFramebuffer(i.READ_FRAMEBUFFER,Tt.__webglMultisampledFramebuffer),e.bindFramebuffer(i.DRAW_FRAMEBUFFER,Tt.__webglFramebuffer);for(let V=0;V<v.length;V++){if(b.resolveDepthBuffer&&(b.depthBuffer&&(J|=i.DEPTH_BUFFER_BIT),b.stencilBuffer&&b.resolveStencilBuffer&&(J|=i.STENCIL_BUFFER_BIT)),dt){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Tt.__webglColorRenderbuffer[V]);const at=n.get(v[V]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,at,0)}i.blitFramebuffer(0,0,F,K,0,0,F,K,J,i.NEAREST),l===!0&&(ue.length=0,R.length=0,ue.push(i.COLOR_ATTACHMENT0+V),b.depthBuffer&&b.resolveDepthBuffer===!1&&(ue.push(j),R.push(j),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,R)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,ue))}if(e.bindFramebuffer(i.READ_FRAMEBUFFER,null),e.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),dt)for(let V=0;V<v.length;V++){e.bindFramebuffer(i.FRAMEBUFFER,Tt.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+V,i.RENDERBUFFER,Tt.__webglColorRenderbuffer[V]);const at=n.get(v[V]).__webglTexture;e.bindFramebuffer(i.FRAMEBUFFER,Tt.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+V,i.TEXTURE_2D,at,0)}e.bindFramebuffer(i.DRAW_FRAMEBUFFER,Tt.__webglMultisampledFramebuffer)}else if(b.depthBuffer&&b.resolveDepthBuffer===!1&&l){const v=b.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[v])}}}function qt(b){return Math.min(s.maxSamples,b.samples)}function jt(b){const v=n.get(b);return b.samples>0&&t.has("WEBGL_multisampled_render_to_texture")===!0&&v.__useRenderToTexture!==!1}function At(b){const v=a.render.frame;h.get(b)!==v&&(h.set(b,v),b.update())}function le(b,v){const F=b.colorSpace,K=b.format,J=b.type;return b.isCompressedTexture===!0||b.isVideoTexture===!0||F!==Ji&&F!==Vn&&(Qt.getTransfer(F)===se?(K!==dn||J!==Ln)&&console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):console.error("THREE.WebGLTextures: Unsupported texture color space:",F)),v}function Rt(b){return typeof HTMLImageElement<"u"&&b instanceof HTMLImageElement?(c.width=b.naturalWidth||b.width,c.height=b.naturalHeight||b.height):typeof VideoFrame<"u"&&b instanceof VideoFrame?(c.width=b.displayWidth,c.height=b.displayHeight):(c.width=b.width,c.height=b.height),c}this.allocateTextureUnit=k,this.resetTextureUnits=W,this.setTexture2D=Q,this.setTexture2DArray=X,this.setTexture3D=tt,this.setTextureCube=H,this.rebindTextures=Vt,this.setupRenderTarget=ie,this.updateRenderTargetMipmap=Wt,this.updateMultisampleRenderTarget=ye,this.setupDepthRenderbuffer=Lt,this.setupFrameBufferTexture=_t,this.useMultisampledRTT=jt}function Qm(i,t){function e(n,s=Vn){let r;const a=Qt.getTransfer(s);if(n===Ln)return i.UNSIGNED_BYTE;if(n===Mo)return i.UNSIGNED_SHORT_4_4_4_4;if(n===So)return i.UNSIGNED_SHORT_5_5_5_1;if(n===rc)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===ic)return i.BYTE;if(n===sc)return i.SHORT;if(n===Ms)return i.UNSIGNED_SHORT;if(n===xo)return i.INT;if(n===li)return i.UNSIGNED_INT;if(n===xn)return i.FLOAT;if(n===Pn)return i.HALF_FLOAT;if(n===ac)return i.ALPHA;if(n===oc)return i.RGB;if(n===dn)return i.RGBA;if(n===lc)return i.LUMINANCE;if(n===cc)return i.LUMINANCE_ALPHA;if(n===Gi)return i.DEPTH_COMPONENT;if(n===$i)return i.DEPTH_STENCIL;if(n===yo)return i.RED;if(n===Eo)return i.RED_INTEGER;if(n===hc)return i.RG;if(n===bo)return i.RG_INTEGER;if(n===To)return i.RGBA_INTEGER;if(n===ur||n===dr||n===fr||n===pr)if(a===se)if(r=t.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===ur)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===dr)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===fr)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===pr)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=t.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===ur)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===dr)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===fr)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===pr)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===Na||n===Fa||n===Oa||n===Ba)if(r=t.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===Na)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===Fa)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===Oa)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===Ba)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===za||n===ka||n===Ha)if(r=t.get("WEBGL_compressed_texture_etc"),r!==null){if(n===za||n===ka)return a===se?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===Ha)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(n===Va||n===Ga||n===Wa||n===Xa||n===Ya||n===qa||n===ja||n===Za||n===Ka||n===$a||n===Ja||n===Qa||n===to||n===eo)if(r=t.get("WEBGL_compressed_texture_astc"),r!==null){if(n===Va)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Ga)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===Wa)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Xa)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Ya)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===qa)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===ja)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Za)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Ka)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===$a)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===Ja)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===Qa)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===to)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===eo)return a===se?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===mr||n===no||n===io)if(r=t.get("EXT_texture_compression_bptc"),r!==null){if(n===mr)return a===se?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===no)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===io)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===uc||n===so||n===ro||n===ao)if(r=t.get("EXT_texture_compression_rgtc"),r!==null){if(n===mr)return r.COMPRESSED_RED_RGTC1_EXT;if(n===so)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===ro)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===ao)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===Ki?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:e}}const tg={type:"move"};class ga{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new zi,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new zi,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new P,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new P),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new zi,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new P,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new P),this._grip}dispatchEvent(t){return this._targetRay!==null&&this._targetRay.dispatchEvent(t),this._grip!==null&&this._grip.dispatchEvent(t),this._hand!==null&&this._hand.dispatchEvent(t),this}connect(t){if(t&&t.hand){const e=this._hand;if(e)for(const n of t.hand.values())this._getHandJoint(e,n)}return this.dispatchEvent({type:"connected",data:t}),this}disconnect(t){return this.dispatchEvent({type:"disconnected",data:t}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(t,e,n){let s=null,r=null,a=null;const o=this._targetRay,l=this._grip,c=this._hand;if(t&&e.session.visibilityState!=="visible-blurred"){if(c&&t.hand){a=!0;for(const _ of t.hand.values()){const m=e.getJointPose(_,n),f=this._getHandJoint(c,_);m!==null&&(f.matrix.fromArray(m.transform.matrix),f.matrix.decompose(f.position,f.rotation,f.scale),f.matrixWorldNeedsUpdate=!0,f.jointRadius=m.radius),f.visible=m!==null}const h=c.joints["index-finger-tip"],d=c.joints["thumb-tip"],p=h.position.distanceTo(d.position),u=.02,g=.005;c.inputState.pinching&&p>u+g?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:t.handedness,target:this})):!c.inputState.pinching&&p<=u-g&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:t.handedness,target:this}))}else l!==null&&t.gripSpace&&(r=e.getPose(t.gripSpace,n),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1));o!==null&&(s=e.getPose(t.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(o.matrix.fromArray(s.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,s.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(s.linearVelocity)):o.hasLinearVelocity=!1,s.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(s.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(tg)))}return o!==null&&(o.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=a!==null),this}_getHandJoint(t,e){if(t.joints[e.jointName]===void 0){const n=new zi;n.matrixAutoUpdate=!1,n.visible=!1,t.joints[e.jointName]=n,t.add(n)}return t.joints[e.jointName]}}const eg=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,ng=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class ig{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(t,e,n){if(this.texture===null){const s=new Pe,r=t.properties.get(s);r.__webglTexture=e.texture,(e.depthNear!==n.depthNear||e.depthFar!==n.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=s}}getMesh(t){if(this.texture!==null&&this.mesh===null){const e=t.cameras[0].viewport,n=new He({vertexShader:eg,fragmentShader:ng,uniforms:{depthColor:{value:this.texture},depthWidth:{value:e.z},depthHeight:{value:e.w}}});this.mesh=new be(new ws(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class sg extends hi{constructor(t,e){super();const n=this;let s=null,r=1,a=null,o="local-floor",l=1,c=null,h=null,d=null,p=null,u=null,g=null;const _=new ig,m=e.getContextAttributes();let f=null,T=null;const E=[],y=[],D=new St;let w=null;const C=new $e;C.viewport=new oe;const I=new $e;I.viewport=new oe;const S=[C,I],M=new Su;let A=null,W=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(Y){let nt=E[Y];return nt===void 0&&(nt=new ga,E[Y]=nt),nt.getTargetRaySpace()},this.getControllerGrip=function(Y){let nt=E[Y];return nt===void 0&&(nt=new ga,E[Y]=nt),nt.getGripSpace()},this.getHand=function(Y){let nt=E[Y];return nt===void 0&&(nt=new ga,E[Y]=nt),nt.getHandSpace()};function k(Y){const nt=y.indexOf(Y.inputSource);if(nt===-1)return;const _t=E[nt];_t!==void 0&&(_t.update(Y.inputSource,Y.frame,c||a),_t.dispatchEvent({type:Y.type,data:Y.inputSource}))}function q(){s.removeEventListener("select",k),s.removeEventListener("selectstart",k),s.removeEventListener("selectend",k),s.removeEventListener("squeeze",k),s.removeEventListener("squeezestart",k),s.removeEventListener("squeezeend",k),s.removeEventListener("end",q),s.removeEventListener("inputsourceschange",Q);for(let Y=0;Y<E.length;Y++){const nt=y[Y];nt!==null&&(y[Y]=null,E[Y].disconnect(nt))}A=null,W=null,_.reset(),t.setRenderTarget(f),u=null,p=null,d=null,s=null,T=null,Xt.stop(),n.isPresenting=!1,t.setPixelRatio(w),t.setSize(D.width,D.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(Y){r=Y,n.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(Y){o=Y,n.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||a},this.setReferenceSpace=function(Y){c=Y},this.getBaseLayer=function(){return p!==null?p:u},this.getBinding=function(){return d},this.getFrame=function(){return g},this.getSession=function(){return s},this.setSession=async function(Y){if(s=Y,s!==null){if(f=t.getRenderTarget(),s.addEventListener("select",k),s.addEventListener("selectstart",k),s.addEventListener("selectend",k),s.addEventListener("squeeze",k),s.addEventListener("squeezestart",k),s.addEventListener("squeezeend",k),s.addEventListener("end",q),s.addEventListener("inputsourceschange",Q),m.xrCompatible!==!0&&await e.makeXRCompatible(),w=t.getPixelRatio(),t.getSize(D),s.enabledFeatures!==void 0&&s.enabledFeatures.includes("layers")){let _t=null,lt=null,Ct=null;m.depth&&(Ct=m.stencil?e.DEPTH24_STENCIL8:e.DEPTH_COMPONENT24,_t=m.stencil?$i:Gi,lt=m.stencil?Ki:li);const Lt={colorFormat:e.RGBA8,depthFormat:Ct,scaleFactor:r};d=new XRWebGLBinding(s,e),p=d.createProjectionLayer(Lt),s.updateRenderState({layers:[p]}),t.setPixelRatio(1),t.setSize(p.textureWidth,p.textureHeight,!1),T=new fn(p.textureWidth,p.textureHeight,{format:dn,type:Ln,depthTexture:new Tc(p.textureWidth,p.textureHeight,lt,void 0,void 0,void 0,void 0,void 0,void 0,_t),stencilBuffer:m.stencil,colorSpace:t.outputColorSpace,samples:m.antialias?4:0,resolveDepthBuffer:p.ignoreDepthValues===!1})}else{const _t={antialias:m.antialias,alpha:!0,depth:m.depth,stencil:m.stencil,framebufferScaleFactor:r};u=new XRWebGLLayer(s,e,_t),s.updateRenderState({baseLayer:u}),t.setPixelRatio(1),t.setSize(u.framebufferWidth,u.framebufferHeight,!1),T=new fn(u.framebufferWidth,u.framebufferHeight,{format:dn,type:Ln,colorSpace:t.outputColorSpace,stencilBuffer:m.stencil})}T.isXRRenderTarget=!0,this.setFoveation(l),c=null,a=await s.requestReferenceSpace(o),Xt.setContext(s),Xt.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return _.getDepthTexture()};function Q(Y){for(let nt=0;nt<Y.removed.length;nt++){const _t=Y.removed[nt],lt=y.indexOf(_t);lt>=0&&(y[lt]=null,E[lt].disconnect(_t))}for(let nt=0;nt<Y.added.length;nt++){const _t=Y.added[nt];let lt=y.indexOf(_t);if(lt===-1){for(let Lt=0;Lt<E.length;Lt++)if(Lt>=y.length){y.push(_t),lt=Lt;break}else if(y[Lt]===null){y[Lt]=_t,lt=Lt;break}if(lt===-1)break}const Ct=E[lt];Ct&&Ct.connect(_t)}}const X=new P,tt=new P;function H(Y,nt,_t){X.setFromMatrixPosition(nt.matrixWorld),tt.setFromMatrixPosition(_t.matrixWorld);const lt=X.distanceTo(tt),Ct=nt.projectionMatrix.elements,Lt=_t.projectionMatrix.elements,Vt=Ct[14]/(Ct[10]-1),ie=Ct[14]/(Ct[10]+1),Wt=(Ct[9]+1)/Ct[5],ue=(Ct[9]-1)/Ct[5],R=(Ct[8]-1)/Ct[0],ye=(Lt[8]+1)/Lt[0],qt=Vt*R,jt=Vt*ye,At=lt/(-R+ye),le=At*-R;if(nt.matrixWorld.decompose(Y.position,Y.quaternion,Y.scale),Y.translateX(le),Y.translateZ(At),Y.matrixWorld.compose(Y.position,Y.quaternion,Y.scale),Y.matrixWorldInverse.copy(Y.matrixWorld).invert(),Ct[10]===-1)Y.projectionMatrix.copy(nt.projectionMatrix),Y.projectionMatrixInverse.copy(nt.projectionMatrixInverse);else{const Rt=Vt+At,b=ie+At,v=qt-le,F=jt+(lt-le),K=Wt*ie/b*Rt,J=ue*ie/b*Rt;Y.projectionMatrix.makePerspective(v,F,K,J,Rt,b),Y.projectionMatrixInverse.copy(Y.projectionMatrix).invert()}}function st(Y,nt){nt===null?Y.matrixWorld.copy(Y.matrix):Y.matrixWorld.multiplyMatrices(nt.matrixWorld,Y.matrix),Y.matrixWorldInverse.copy(Y.matrixWorld).invert()}this.updateCamera=function(Y){if(s===null)return;let nt=Y.near,_t=Y.far;_.texture!==null&&(_.depthNear>0&&(nt=_.depthNear),_.depthFar>0&&(_t=_.depthFar)),M.near=I.near=C.near=nt,M.far=I.far=C.far=_t,(A!==M.near||W!==M.far)&&(s.updateRenderState({depthNear:M.near,depthFar:M.far}),A=M.near,W=M.far),C.layers.mask=Y.layers.mask|2,I.layers.mask=Y.layers.mask|4,M.layers.mask=C.layers.mask|I.layers.mask;const lt=Y.parent,Ct=M.cameras;st(M,lt);for(let Lt=0;Lt<Ct.length;Lt++)st(Ct[Lt],lt);Ct.length===2?H(M,C,I):M.projectionMatrix.copy(C.projectionMatrix),gt(Y,M,lt)};function gt(Y,nt,_t){_t===null?Y.matrix.copy(nt.matrixWorld):(Y.matrix.copy(_t.matrixWorld),Y.matrix.invert(),Y.matrix.multiply(nt.matrixWorld)),Y.matrix.decompose(Y.position,Y.quaternion,Y.scale),Y.updateMatrixWorld(!0),Y.projectionMatrix.copy(nt.projectionMatrix),Y.projectionMatrixInverse.copy(nt.projectionMatrixInverse),Y.isPerspectiveCamera&&(Y.fov=lo*2*Math.atan(1/Y.projectionMatrix.elements[5]),Y.zoom=1)}this.getCamera=function(){return M},this.getFoveation=function(){if(!(p===null&&u===null))return l},this.setFoveation=function(Y){l=Y,p!==null&&(p.fixedFoveation=Y),u!==null&&u.fixedFoveation!==void 0&&(u.fixedFoveation=Y)},this.hasDepthSensing=function(){return _.texture!==null},this.getDepthSensingMesh=function(){return _.getMesh(M)};let Et=null;function Ft(Y,nt){if(h=nt.getViewerPose(c||a),g=nt,h!==null){const _t=h.views;u!==null&&(t.setRenderTargetFramebuffer(T,u.framebuffer),t.setRenderTarget(T));let lt=!1;_t.length!==M.cameras.length&&(M.cameras.length=0,lt=!0);for(let Lt=0;Lt<_t.length;Lt++){const Vt=_t[Lt];let ie=null;if(u!==null)ie=u.getViewport(Vt);else{const ue=d.getViewSubImage(p,Vt);ie=ue.viewport,Lt===0&&(t.setRenderTargetTextures(T,ue.colorTexture,p.ignoreDepthValues?void 0:ue.depthStencilTexture),t.setRenderTarget(T))}let Wt=S[Lt];Wt===void 0&&(Wt=new $e,Wt.layers.enable(Lt),Wt.viewport=new oe,S[Lt]=Wt),Wt.matrix.fromArray(Vt.transform.matrix),Wt.matrix.decompose(Wt.position,Wt.quaternion,Wt.scale),Wt.projectionMatrix.fromArray(Vt.projectionMatrix),Wt.projectionMatrixInverse.copy(Wt.projectionMatrix).invert(),Wt.viewport.set(ie.x,ie.y,ie.width,ie.height),Lt===0&&(M.matrix.copy(Wt.matrix),M.matrix.decompose(M.position,M.quaternion,M.scale)),lt===!0&&M.cameras.push(Wt)}const Ct=s.enabledFeatures;if(Ct&&Ct.includes("depth-sensing")){const Lt=d.getDepthInformation(_t[0]);Lt&&Lt.isValid&&Lt.texture&&_.init(t,Lt,s.renderState)}}for(let _t=0;_t<E.length;_t++){const lt=y[_t],Ct=E[_t];lt!==null&&Ct!==void 0&&Ct.update(lt,nt,c||a)}Et&&Et(Y,nt),nt.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:nt}),g=null}const Xt=new Rc;Xt.setAnimationLoop(Ft),this.setAnimationLoop=function(Y){Et=Y},this.dispose=function(){}}}const ti=new Mn,rg=new ne;function ag(i,t){function e(m,f){m.matrixAutoUpdate===!0&&m.updateMatrix(),f.value.copy(m.matrix)}function n(m,f){f.color.getRGB(m.fogColor.value,Mc(i)),f.isFog?(m.fogNear.value=f.near,m.fogFar.value=f.far):f.isFogExp2&&(m.fogDensity.value=f.density)}function s(m,f,T,E,y){f.isMeshBasicMaterial||f.isMeshLambertMaterial?r(m,f):f.isMeshToonMaterial?(r(m,f),d(m,f)):f.isMeshPhongMaterial?(r(m,f),h(m,f)):f.isMeshStandardMaterial?(r(m,f),p(m,f),f.isMeshPhysicalMaterial&&u(m,f,y)):f.isMeshMatcapMaterial?(r(m,f),g(m,f)):f.isMeshDepthMaterial?r(m,f):f.isMeshDistanceMaterial?(r(m,f),_(m,f)):f.isMeshNormalMaterial?r(m,f):f.isLineBasicMaterial?(a(m,f),f.isLineDashedMaterial&&o(m,f)):f.isPointsMaterial?l(m,f,T,E):f.isSpriteMaterial?c(m,f):f.isShadowMaterial?(m.color.value.copy(f.color),m.opacity.value=f.opacity):f.isShaderMaterial&&(f.uniformsNeedUpdate=!1)}function r(m,f){m.opacity.value=f.opacity,f.color&&m.diffuse.value.copy(f.color),f.emissive&&m.emissive.value.copy(f.emissive).multiplyScalar(f.emissiveIntensity),f.map&&(m.map.value=f.map,e(f.map,m.mapTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,e(f.alphaMap,m.alphaMapTransform)),f.bumpMap&&(m.bumpMap.value=f.bumpMap,e(f.bumpMap,m.bumpMapTransform),m.bumpScale.value=f.bumpScale,f.side===Xe&&(m.bumpScale.value*=-1)),f.normalMap&&(m.normalMap.value=f.normalMap,e(f.normalMap,m.normalMapTransform),m.normalScale.value.copy(f.normalScale),f.side===Xe&&m.normalScale.value.negate()),f.displacementMap&&(m.displacementMap.value=f.displacementMap,e(f.displacementMap,m.displacementMapTransform),m.displacementScale.value=f.displacementScale,m.displacementBias.value=f.displacementBias),f.emissiveMap&&(m.emissiveMap.value=f.emissiveMap,e(f.emissiveMap,m.emissiveMapTransform)),f.specularMap&&(m.specularMap.value=f.specularMap,e(f.specularMap,m.specularMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest);const T=t.get(f),E=T.envMap,y=T.envMapRotation;E&&(m.envMap.value=E,ti.copy(y),ti.x*=-1,ti.y*=-1,ti.z*=-1,E.isCubeTexture&&E.isRenderTargetTexture===!1&&(ti.y*=-1,ti.z*=-1),m.envMapRotation.value.setFromMatrix4(rg.makeRotationFromEuler(ti)),m.flipEnvMap.value=E.isCubeTexture&&E.isRenderTargetTexture===!1?-1:1,m.reflectivity.value=f.reflectivity,m.ior.value=f.ior,m.refractionRatio.value=f.refractionRatio),f.lightMap&&(m.lightMap.value=f.lightMap,m.lightMapIntensity.value=f.lightMapIntensity,e(f.lightMap,m.lightMapTransform)),f.aoMap&&(m.aoMap.value=f.aoMap,m.aoMapIntensity.value=f.aoMapIntensity,e(f.aoMap,m.aoMapTransform))}function a(m,f){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,f.map&&(m.map.value=f.map,e(f.map,m.mapTransform))}function o(m,f){m.dashSize.value=f.dashSize,m.totalSize.value=f.dashSize+f.gapSize,m.scale.value=f.scale}function l(m,f,T,E){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,m.size.value=f.size*T,m.scale.value=E*.5,f.map&&(m.map.value=f.map,e(f.map,m.uvTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,e(f.alphaMap,m.alphaMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest)}function c(m,f){m.diffuse.value.copy(f.color),m.opacity.value=f.opacity,m.rotation.value=f.rotation,f.map&&(m.map.value=f.map,e(f.map,m.mapTransform)),f.alphaMap&&(m.alphaMap.value=f.alphaMap,e(f.alphaMap,m.alphaMapTransform)),f.alphaTest>0&&(m.alphaTest.value=f.alphaTest)}function h(m,f){m.specular.value.copy(f.specular),m.shininess.value=Math.max(f.shininess,1e-4)}function d(m,f){f.gradientMap&&(m.gradientMap.value=f.gradientMap)}function p(m,f){m.metalness.value=f.metalness,f.metalnessMap&&(m.metalnessMap.value=f.metalnessMap,e(f.metalnessMap,m.metalnessMapTransform)),m.roughness.value=f.roughness,f.roughnessMap&&(m.roughnessMap.value=f.roughnessMap,e(f.roughnessMap,m.roughnessMapTransform)),f.envMap&&(m.envMapIntensity.value=f.envMapIntensity)}function u(m,f,T){m.ior.value=f.ior,f.sheen>0&&(m.sheenColor.value.copy(f.sheenColor).multiplyScalar(f.sheen),m.sheenRoughness.value=f.sheenRoughness,f.sheenColorMap&&(m.sheenColorMap.value=f.sheenColorMap,e(f.sheenColorMap,m.sheenColorMapTransform)),f.sheenRoughnessMap&&(m.sheenRoughnessMap.value=f.sheenRoughnessMap,e(f.sheenRoughnessMap,m.sheenRoughnessMapTransform))),f.clearcoat>0&&(m.clearcoat.value=f.clearcoat,m.clearcoatRoughness.value=f.clearcoatRoughness,f.clearcoatMap&&(m.clearcoatMap.value=f.clearcoatMap,e(f.clearcoatMap,m.clearcoatMapTransform)),f.clearcoatRoughnessMap&&(m.clearcoatRoughnessMap.value=f.clearcoatRoughnessMap,e(f.clearcoatRoughnessMap,m.clearcoatRoughnessMapTransform)),f.clearcoatNormalMap&&(m.clearcoatNormalMap.value=f.clearcoatNormalMap,e(f.clearcoatNormalMap,m.clearcoatNormalMapTransform),m.clearcoatNormalScale.value.copy(f.clearcoatNormalScale),f.side===Xe&&m.clearcoatNormalScale.value.negate())),f.dispersion>0&&(m.dispersion.value=f.dispersion),f.iridescence>0&&(m.iridescence.value=f.iridescence,m.iridescenceIOR.value=f.iridescenceIOR,m.iridescenceThicknessMinimum.value=f.iridescenceThicknessRange[0],m.iridescenceThicknessMaximum.value=f.iridescenceThicknessRange[1],f.iridescenceMap&&(m.iridescenceMap.value=f.iridescenceMap,e(f.iridescenceMap,m.iridescenceMapTransform)),f.iridescenceThicknessMap&&(m.iridescenceThicknessMap.value=f.iridescenceThicknessMap,e(f.iridescenceThicknessMap,m.iridescenceThicknessMapTransform))),f.transmission>0&&(m.transmission.value=f.transmission,m.transmissionSamplerMap.value=T.texture,m.transmissionSamplerSize.value.set(T.width,T.height),f.transmissionMap&&(m.transmissionMap.value=f.transmissionMap,e(f.transmissionMap,m.transmissionMapTransform)),m.thickness.value=f.thickness,f.thicknessMap&&(m.thicknessMap.value=f.thicknessMap,e(f.thicknessMap,m.thicknessMapTransform)),m.attenuationDistance.value=f.attenuationDistance,m.attenuationColor.value.copy(f.attenuationColor)),f.anisotropy>0&&(m.anisotropyVector.value.set(f.anisotropy*Math.cos(f.anisotropyRotation),f.anisotropy*Math.sin(f.anisotropyRotation)),f.anisotropyMap&&(m.anisotropyMap.value=f.anisotropyMap,e(f.anisotropyMap,m.anisotropyMapTransform))),m.specularIntensity.value=f.specularIntensity,m.specularColor.value.copy(f.specularColor),f.specularColorMap&&(m.specularColorMap.value=f.specularColorMap,e(f.specularColorMap,m.specularColorMapTransform)),f.specularIntensityMap&&(m.specularIntensityMap.value=f.specularIntensityMap,e(f.specularIntensityMap,m.specularIntensityMapTransform))}function g(m,f){f.matcap&&(m.matcap.value=f.matcap)}function _(m,f){const T=t.get(f).light;m.referencePosition.value.setFromMatrixPosition(T.matrixWorld),m.nearDistance.value=T.shadow.camera.near,m.farDistance.value=T.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function og(i,t,e,n){let s={},r={},a=[];const o=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function l(T,E){const y=E.program;n.uniformBlockBinding(T,y)}function c(T,E){let y=s[T.id];y===void 0&&(g(T),y=h(T),s[T.id]=y,T.addEventListener("dispose",m));const D=E.program;n.updateUBOMapping(T,D);const w=t.render.frame;r[T.id]!==w&&(p(T),r[T.id]=w)}function h(T){const E=d();T.__bindingPointIndex=E;const y=i.createBuffer(),D=T.__size,w=T.usage;return i.bindBuffer(i.UNIFORM_BUFFER,y),i.bufferData(i.UNIFORM_BUFFER,D,w),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,E,y),y}function d(){for(let T=0;T<o;T++)if(a.indexOf(T)===-1)return a.push(T),T;return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function p(T){const E=s[T.id],y=T.uniforms,D=T.__cache;i.bindBuffer(i.UNIFORM_BUFFER,E);for(let w=0,C=y.length;w<C;w++){const I=Array.isArray(y[w])?y[w]:[y[w]];for(let S=0,M=I.length;S<M;S++){const A=I[S];if(u(A,w,S,D)===!0){const W=A.__offset,k=Array.isArray(A.value)?A.value:[A.value];let q=0;for(let Q=0;Q<k.length;Q++){const X=k[Q],tt=_(X);typeof X=="number"||typeof X=="boolean"?(A.__data[0]=X,i.bufferSubData(i.UNIFORM_BUFFER,W+q,A.__data)):X.isMatrix3?(A.__data[0]=X.elements[0],A.__data[1]=X.elements[1],A.__data[2]=X.elements[2],A.__data[3]=0,A.__data[4]=X.elements[3],A.__data[5]=X.elements[4],A.__data[6]=X.elements[5],A.__data[7]=0,A.__data[8]=X.elements[6],A.__data[9]=X.elements[7],A.__data[10]=X.elements[8],A.__data[11]=0):(X.toArray(A.__data,q),q+=tt.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,W,A.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function u(T,E,y,D){const w=T.value,C=E+"_"+y;if(D[C]===void 0)return typeof w=="number"||typeof w=="boolean"?D[C]=w:D[C]=w.clone(),!0;{const I=D[C];if(typeof w=="number"||typeof w=="boolean"){if(I!==w)return D[C]=w,!0}else if(I.equals(w)===!1)return I.copy(w),!0}return!1}function g(T){const E=T.uniforms;let y=0;const D=16;for(let C=0,I=E.length;C<I;C++){const S=Array.isArray(E[C])?E[C]:[E[C]];for(let M=0,A=S.length;M<A;M++){const W=S[M],k=Array.isArray(W.value)?W.value:[W.value];for(let q=0,Q=k.length;q<Q;q++){const X=k[q],tt=_(X),H=y%D,st=H%tt.boundary,gt=H+st;y+=st,gt!==0&&D-gt<tt.storage&&(y+=D-gt),W.__data=new Float32Array(tt.storage/Float32Array.BYTES_PER_ELEMENT),W.__offset=y,y+=tt.storage}}}const w=y%D;return w>0&&(y+=D-w),T.__size=y,T.__cache={},this}function _(T){const E={boundary:0,storage:0};return typeof T=="number"||typeof T=="boolean"?(E.boundary=4,E.storage=4):T.isVector2?(E.boundary=8,E.storage=8):T.isVector3||T.isColor?(E.boundary=16,E.storage=12):T.isVector4?(E.boundary=16,E.storage=16):T.isMatrix3?(E.boundary=48,E.storage=48):T.isMatrix4?(E.boundary=64,E.storage=64):T.isTexture?console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group."):console.warn("THREE.WebGLRenderer: Unsupported uniform value type.",T),E}function m(T){const E=T.target;E.removeEventListener("dispose",m);const y=a.indexOf(E.__bindingPointIndex);a.splice(y,1),i.deleteBuffer(s[E.id]),delete s[E.id],delete r[E.id]}function f(){for(const T in s)i.deleteBuffer(s[T]);a=[],s={},r={}}return{bind:l,update:c,dispose:f}}class lg{constructor(t={}){const{canvas:e=Oh(),context:n=null,depth:s=!0,stencil:r=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:h="default",failIfMajorPerformanceCaveat:d=!1,reverseDepthBuffer:p=!1}=t;this.isWebGLRenderer=!0;let u;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");u=n.getContextAttributes().alpha}else u=a;const g=new Uint32Array(4),_=new Int32Array(4);let m=null,f=null;const T=[],E=[];this.domElement=e,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this._outputColorSpace=an,this.toneMapping=Wn,this.toneMappingExposure=1;const y=this;let D=!1,w=0,C=0,I=null,S=-1,M=null;const A=new oe,W=new oe;let k=null;const q=new ot(0);let Q=0,X=e.width,tt=e.height,H=1,st=null,gt=null;const Et=new oe(0,0,X,tt),Ft=new oe(0,0,X,tt);let Xt=!1;const Y=new Ao;let nt=!1,_t=!1;this.transmissionResolutionScale=1;const lt=new ne,Ct=new ne,Lt=new P,Vt=new oe,ie={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let Wt=!1;function ue(){return I===null?H:1}let R=n;function ye(x,U){return e.getContext(x,U)}try{const x={alpha:!0,depth:s,stencil:r,antialias:o,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:h,failIfMajorPerformanceCaveat:d};if("setAttribute"in e&&e.setAttribute("data-engine",`three.js r${vo}`),e.addEventListener("webglcontextlost",$,!1),e.addEventListener("webglcontextrestored",ft,!1),e.addEventListener("webglcontextcreationerror",ut,!1),R===null){const U="webgl2";if(R=ye(U,x),R===null)throw ye(U)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(x){throw console.error("THREE.WebGLRenderer: "+x.message),x}let qt,jt,At,le,Rt,b,v,F,K,J,j,Tt,dt,V,at,Z,it,pt,wt,ht,Bt,Ut,Zt,L;function rt(){qt=new _p(R),qt.init(),Ut=new Qm(R,qt),jt=new up(R,qt,t,Ut),At=new $m(R,qt),jt.reverseDepthBuffer&&p&&At.buffers.depth.setReversed(!0),le=new Mp(R),Rt=new Bm,b=new Jm(R,qt,At,Rt,jt,Ut,le),v=new fp(y),F=new gp(y),K=new wu(R),Zt=new cp(R,K),J=new vp(R,K,le,Zt),j=new yp(R,J,K,le),wt=new Sp(R,jt,b),Z=new dp(Rt),Tt=new Om(y,v,F,qt,jt,Zt,Z),dt=new ag(y,Rt),V=new km,at=new Ym(qt),pt=new lp(y,v,F,At,j,u,l),it=new Zm(y,j,jt),L=new og(R,le,jt,At),ht=new hp(R,qt,le),Bt=new xp(R,qt,le),le.programs=Tt.programs,y.capabilities=jt,y.extensions=qt,y.properties=Rt,y.renderLists=V,y.shadowMap=it,y.state=At,y.info=le}rt();const G=new sg(y,R);this.xr=G,this.getContext=function(){return R},this.getContextAttributes=function(){return R.getContextAttributes()},this.forceContextLoss=function(){const x=qt.get("WEBGL_lose_context");x&&x.loseContext()},this.forceContextRestore=function(){const x=qt.get("WEBGL_lose_context");x&&x.restoreContext()},this.getPixelRatio=function(){return H},this.setPixelRatio=function(x){x!==void 0&&(H=x,this.setSize(X,tt,!1))},this.getSize=function(x){return x.set(X,tt)},this.setSize=function(x,U,O=!0){if(G.isPresenting){console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");return}X=x,tt=U,e.width=Math.floor(x*H),e.height=Math.floor(U*H),O===!0&&(e.style.width=x+"px",e.style.height=U+"px"),this.setViewport(0,0,x,U)},this.getDrawingBufferSize=function(x){return x.set(X*H,tt*H).floor()},this.setDrawingBufferSize=function(x,U,O){X=x,tt=U,H=O,e.width=Math.floor(x*O),e.height=Math.floor(U*O),this.setViewport(0,0,x,U)},this.getCurrentViewport=function(x){return x.copy(A)},this.getViewport=function(x){return x.copy(Et)},this.setViewport=function(x,U,O,z){x.isVector4?Et.set(x.x,x.y,x.z,x.w):Et.set(x,U,O,z),At.viewport(A.copy(Et).multiplyScalar(H).round())},this.getScissor=function(x){return x.copy(Ft)},this.setScissor=function(x,U,O,z){x.isVector4?Ft.set(x.x,x.y,x.z,x.w):Ft.set(x,U,O,z),At.scissor(W.copy(Ft).multiplyScalar(H).round())},this.getScissorTest=function(){return Xt},this.setScissorTest=function(x){At.setScissorTest(Xt=x)},this.setOpaqueSort=function(x){st=x},this.setTransparentSort=function(x){gt=x},this.getClearColor=function(x){return x.copy(pt.getClearColor())},this.setClearColor=function(){pt.setClearColor.apply(pt,arguments)},this.getClearAlpha=function(){return pt.getClearAlpha()},this.setClearAlpha=function(){pt.setClearAlpha.apply(pt,arguments)},this.clear=function(x=!0,U=!0,O=!0){let z=0;if(x){let N=!1;if(I!==null){const et=I.texture.format;N=et===To||et===bo||et===Eo}if(N){const et=I.texture.type,mt=et===Ln||et===li||et===Ms||et===Ki||et===Mo||et===So,vt=pt.getClearColor(),Mt=pt.getClearAlpha(),It=vt.r,Nt=vt.g,Pt=vt.b;mt?(g[0]=It,g[1]=Nt,g[2]=Pt,g[3]=Mt,R.clearBufferuiv(R.COLOR,0,g)):(_[0]=It,_[1]=Nt,_[2]=Pt,_[3]=Mt,R.clearBufferiv(R.COLOR,0,_))}else z|=R.COLOR_BUFFER_BIT}U&&(z|=R.DEPTH_BUFFER_BIT),O&&(z|=R.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),R.clear(z)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){e.removeEventListener("webglcontextlost",$,!1),e.removeEventListener("webglcontextrestored",ft,!1),e.removeEventListener("webglcontextcreationerror",ut,!1),pt.dispose(),V.dispose(),at.dispose(),Rt.dispose(),v.dispose(),F.dispose(),j.dispose(),Zt.dispose(),L.dispose(),Tt.dispose(),G.dispose(),G.removeEventListener("sessionstart",Rs),G.removeEventListener("sessionend",fi),pn.stop()};function $(x){x.preventDefault(),console.log("THREE.WebGLRenderer: Context Lost."),D=!0}function ft(){console.log("THREE.WebGLRenderer: Context Restored."),D=!1;const x=le.autoReset,U=it.enabled,O=it.autoUpdate,z=it.needsUpdate,N=it.type;rt(),le.autoReset=x,it.enabled=U,it.autoUpdate=O,it.needsUpdate=z,it.type=N}function ut(x){console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ",x.statusMessage)}function Ot(x){const U=x.target;U.removeEventListener("dispose",Ot),ce(U)}function ce(x){Ae(x),Rt.remove(x)}function Ae(x){const U=Rt.get(x).programs;U!==void 0&&(U.forEach(function(O){Tt.releaseProgram(O)}),x.isShaderMaterial&&Tt.releaseShaderCache(x))}this.renderBufferDirect=function(x,U,O,z,N,et){U===null&&(U=ie);const mt=N.isMesh&&N.matrixWorld.determinant()<0,vt=Ds(x,U,O,z,N);At.setMaterial(z,mt);let Mt=O.index,It=1;if(z.wireframe===!0){if(Mt=J.getWireframeAttribute(O),Mt===void 0)return;It=2}const Nt=O.drawRange,Pt=O.attributes.position;let Kt=Nt.start*It,te=(Nt.start+Nt.count)*It;et!==null&&(Kt=Math.max(Kt,et.start*It),te=Math.min(te,(et.start+et.count)*It)),Mt!==null?(Kt=Math.max(Kt,0),te=Math.min(te,Mt.count)):Pt!=null&&(Kt=Math.max(Kt,0),te=Math.min(te,Pt.count));const xe=te-Kt;if(xe<0||xe===1/0)return;Zt.setup(N,z,vt,O,Mt);let pe,Jt=ht;if(Mt!==null&&(pe=K.get(Mt),Jt=Bt,Jt.setIndex(pe)),N.isMesh)z.wireframe===!0?(At.setLineWidth(z.wireframeLinewidth*ue()),Jt.setMode(R.LINES)):Jt.setMode(R.TRIANGLES);else if(N.isLine){let Dt=z.linewidth;Dt===void 0&&(Dt=1),At.setLineWidth(Dt*ue()),N.isLineSegments?Jt.setMode(R.LINES):N.isLineLoop?Jt.setMode(R.LINE_LOOP):Jt.setMode(R.LINE_STRIP)}else N.isPoints?Jt.setMode(R.POINTS):N.isSprite&&Jt.setMode(R.TRIANGLES);if(N.isBatchedMesh)if(N._multiDrawInstances!==null)Jt.renderMultiDrawInstances(N._multiDrawStarts,N._multiDrawCounts,N._multiDrawCount,N._multiDrawInstances);else if(qt.get("WEBGL_multi_draw"))Jt.renderMultiDraw(N._multiDrawStarts,N._multiDrawCounts,N._multiDrawCount);else{const Dt=N._multiDrawStarts,De=N._multiDrawCounts,ee=N._multiDrawCount,ln=Mt?K.get(Mt).bytesPerElement:1,mi=Rt.get(z).currentProgram.getUniforms();for(let qe=0;qe<ee;qe++)mi.setValue(R,"_gl_DrawID",qe),Jt.render(Dt[qe]/ln,De[qe])}else if(N.isInstancedMesh)Jt.renderInstances(Kt,xe,N.count);else if(O.isInstancedBufferGeometry){const Dt=O._maxInstanceCount!==void 0?O._maxInstanceCount:1/0,De=Math.min(O.instanceCount,Dt);Jt.renderInstances(Kt,xe,De)}else Jt.render(Kt,xe)};function $t(x,U,O){x.transparent===!0&&x.side===gn&&x.forceSinglePass===!1?(x.side=Xe,x.needsUpdate=!0,en(x,U,O),x.side=Yn,x.needsUpdate=!0,en(x,U,O),x.side=gn):en(x,U,O)}this.compile=function(x,U,O=null){O===null&&(O=x),f=at.get(O),f.init(U),E.push(f),O.traverseVisible(function(N){N.isLight&&N.layers.test(U.layers)&&(f.pushLight(N),N.castShadow&&f.pushShadow(N))}),x!==O&&x.traverseVisible(function(N){N.isLight&&N.layers.test(U.layers)&&(f.pushLight(N),N.castShadow&&f.pushShadow(N))}),f.setupLights();const z=new Set;return x.traverse(function(N){if(!(N.isMesh||N.isPoints||N.isLine||N.isSprite))return;const et=N.material;if(et)if(Array.isArray(et))for(let mt=0;mt<et.length;mt++){const vt=et[mt];$t(vt,O,N),z.add(vt)}else $t(et,O,N),z.add(et)}),E.pop(),f=null,z},this.compileAsync=function(x,U,O=null){const z=this.compile(x,U,O);return new Promise(N=>{function et(){if(z.forEach(function(mt){Rt.get(mt).currentProgram.isReady()&&z.delete(mt)}),z.size===0){N(x);return}setTimeout(et,10)}qt.get("KHR_parallel_shader_compile")!==null?et():setTimeout(et,10)})};let Ye=null;function Qe(x){Ye&&Ye(x)}function Rs(){pn.stop()}function fi(){pn.start()}const pn=new Rc;pn.setAnimationLoop(Qe),typeof self<"u"&&pn.setContext(self),this.setAnimationLoop=function(x){Ye=x,G.setAnimationLoop(x),x===null?pn.stop():pn.start()},G.addEventListener("sessionstart",Rs),G.addEventListener("sessionend",fi),this.render=function(x,U){if(U!==void 0&&U.isCamera!==!0){console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(D===!0)return;if(x.matrixWorldAutoUpdate===!0&&x.updateMatrixWorld(),U.parent===null&&U.matrixWorldAutoUpdate===!0&&U.updateMatrixWorld(),G.enabled===!0&&G.isPresenting===!0&&(G.cameraAutoUpdate===!0&&G.updateCamera(U),U=G.getCamera()),x.isScene===!0&&x.onBeforeRender(y,x,U,I),f=at.get(x,E.length),f.init(U),E.push(f),Ct.multiplyMatrices(U.projectionMatrix,U.matrixWorldInverse),Y.setFromProjectionMatrix(Ct),_t=this.localClippingEnabled,nt=Z.init(this.clippingPlanes,_t),m=V.get(x,T.length),m.init(),T.push(m),G.enabled===!0&&G.isPresenting===!0){const et=y.xr.getDepthSensingMesh();et!==null&&ns(et,U,-1/0,y.sortObjects)}ns(x,U,0,y.sortObjects),m.finish(),y.sortObjects===!0&&m.sort(st,gt),Wt=G.enabled===!1||G.isPresenting===!1||G.hasDepthSensing()===!1,Wt&&pt.addToRenderList(m,x),this.info.render.frame++,nt===!0&&Z.beginShadows();const O=f.state.shadowsArray;it.render(O,x,U),nt===!0&&Z.endShadows(),this.info.autoReset===!0&&this.info.reset();const z=m.opaque,N=m.transmissive;if(f.setupLights(),U.isArrayCamera){const et=U.cameras;if(N.length>0)for(let mt=0,vt=et.length;mt<vt;mt++){const Mt=et[mt];Ve(z,N,x,Mt)}Wt&&pt.render(x);for(let mt=0,vt=et.length;mt<vt;mt++){const Mt=et[mt];Cs(m,x,Mt,Mt.viewport)}}else N.length>0&&Ve(z,N,x,U),Wt&&pt.render(x),Cs(m,x,U);I!==null&&C===0&&(b.updateMultisampleRenderTarget(I),b.updateRenderTargetMipmap(I)),x.isScene===!0&&x.onAfterRender(y,x,U),Zt.resetDefaultState(),S=-1,M=null,E.pop(),E.length>0?(f=E[E.length-1],nt===!0&&Z.setGlobalState(y.clippingPlanes,f.state.camera)):f=null,T.pop(),T.length>0?m=T[T.length-1]:m=null};function ns(x,U,O,z){if(x.visible===!1)return;if(x.layers.test(U.layers)){if(x.isGroup)O=x.renderOrder;else if(x.isLOD)x.autoUpdate===!0&&x.update(U);else if(x.isLight)f.pushLight(x),x.castShadow&&f.pushShadow(x);else if(x.isSprite){if(!x.frustumCulled||Y.intersectsSprite(x)){z&&Vt.setFromMatrixPosition(x.matrixWorld).applyMatrix4(Ct);const mt=j.update(x),vt=x.material;vt.visible&&m.push(x,mt,vt,O,Vt.z,null)}}else if((x.isMesh||x.isLine||x.isPoints)&&(!x.frustumCulled||Y.intersectsObject(x))){const mt=j.update(x),vt=x.material;if(z&&(x.boundingSphere!==void 0?(x.boundingSphere===null&&x.computeBoundingSphere(),Vt.copy(x.boundingSphere.center)):(mt.boundingSphere===null&&mt.computeBoundingSphere(),Vt.copy(mt.boundingSphere.center)),Vt.applyMatrix4(x.matrixWorld).applyMatrix4(Ct)),Array.isArray(vt)){const Mt=mt.groups;for(let It=0,Nt=Mt.length;It<Nt;It++){const Pt=Mt[It],Kt=vt[Pt.materialIndex];Kt&&Kt.visible&&m.push(x,mt,Kt,O,Vt.z,Pt)}}else vt.visible&&m.push(x,mt,vt,O,Vt.z,null)}}const et=x.children;for(let mt=0,vt=et.length;mt<vt;mt++)ns(et[mt],U,O,z)}function Cs(x,U,O,z){const N=x.opaque,et=x.transmissive,mt=x.transparent;f.setupLightsView(O),nt===!0&&Z.setGlobalState(y.clippingPlanes,O),z&&At.viewport(A.copy(z)),N.length>0&&Re(N,U,O),et.length>0&&Re(et,U,O),mt.length>0&&Re(mt,U,O),At.buffers.depth.setTest(!0),At.buffers.depth.setMask(!0),At.buffers.color.setMask(!0),At.setPolygonOffset(!1)}function Ve(x,U,O,z){if((O.isScene===!0?O.overrideMaterial:null)!==null)return;f.state.transmissionRenderTarget[z.id]===void 0&&(f.state.transmissionRenderTarget[z.id]=new fn(1,1,{generateMipmaps:!0,type:qt.has("EXT_color_buffer_half_float")||qt.has("EXT_color_buffer_float")?Pn:Ln,minFilter:ai,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:Qt.workingColorSpace}));const et=f.state.transmissionRenderTarget[z.id],mt=z.viewport||A;et.setSize(mt.z*y.transmissionResolutionScale,mt.w*y.transmissionResolutionScale);const vt=y.getRenderTarget();y.setRenderTarget(et),y.getClearColor(q),Q=y.getClearAlpha(),Q<1&&y.setClearColor(16777215,.5),y.clear(),Wt&&pt.render(O);const Mt=y.toneMapping;y.toneMapping=Wn;const It=z.viewport;if(z.viewport!==void 0&&(z.viewport=void 0),f.setupLightsView(z),nt===!0&&Z.setGlobalState(y.clippingPlanes,z),Re(x,O,z),b.updateMultisampleRenderTarget(et),b.updateRenderTargetMipmap(et),qt.has("WEBGL_multisampled_render_to_texture")===!1){let Nt=!1;for(let Pt=0,Kt=U.length;Pt<Kt;Pt++){const te=U[Pt],xe=te.object,pe=te.geometry,Jt=te.material,Dt=te.group;if(Jt.side===gn&&xe.layers.test(z.layers)){const De=Jt.side;Jt.side=Xe,Jt.needsUpdate=!0,tn(xe,O,z,pe,Jt,Dt),Jt.side=De,Jt.needsUpdate=!0,Nt=!0}}Nt===!0&&(b.updateMultisampleRenderTarget(et),b.updateRenderTargetMipmap(et))}y.setRenderTarget(vt),y.setClearColor(q,Q),It!==void 0&&(z.viewport=It),y.toneMapping=Mt}function Re(x,U,O){const z=U.isScene===!0?U.overrideMaterial:null;for(let N=0,et=x.length;N<et;N++){const mt=x[N],vt=mt.object,Mt=mt.geometry,It=z===null?mt.material:z,Nt=mt.group;vt.layers.test(O.layers)&&tn(vt,U,O,Mt,It,Nt)}}function tn(x,U,O,z,N,et){x.onBeforeRender(y,U,O,z,N,et),x.modelViewMatrix.multiplyMatrices(O.matrixWorldInverse,x.matrixWorld),x.normalMatrix.getNormalMatrix(x.modelViewMatrix),N.onBeforeRender(y,U,O,z,x,et),N.transparent===!0&&N.side===gn&&N.forceSinglePass===!1?(N.side=Xe,N.needsUpdate=!0,y.renderBufferDirect(O,U,z,N,x,et),N.side=Yn,N.needsUpdate=!0,y.renderBufferDirect(O,U,z,N,x,et),N.side=gn):y.renderBufferDirect(O,U,z,N,x,et),x.onAfterRender(y,U,O,z,N,et)}function en(x,U,O){U.isScene!==!0&&(U=ie);const z=Rt.get(x),N=f.state.lights,et=f.state.shadowsArray,mt=N.state.version,vt=Tt.getParameters(x,N.state,et,U,O),Mt=Tt.getProgramCacheKey(vt);let It=z.programs;z.environment=x.isMeshStandardMaterial?U.environment:null,z.fog=U.fog,z.envMap=(x.isMeshStandardMaterial?F:v).get(x.envMap||z.environment),z.envMapRotation=z.environment!==null&&x.envMap===null?U.environmentRotation:x.envMapRotation,It===void 0&&(x.addEventListener("dispose",Ot),It=new Map,z.programs=It);let Nt=It.get(Mt);if(Nt!==void 0){if(z.currentProgram===Nt&&z.lightsStateVersion===mt)return Ps(x,vt),Nt}else vt.uniforms=Tt.getUniforms(x),x.onBeforeCompile(vt,y),Nt=Tt.acquireProgram(vt,Mt),It.set(Mt,Nt),z.uniforms=vt.uniforms;const Pt=z.uniforms;return(!x.isShaderMaterial&&!x.isRawShaderMaterial||x.clipping===!0)&&(Pt.clippingPlanes=Z.uniform),Ps(x,vt),z.needsLights=Ls(x),z.lightsStateVersion=mt,z.needsLights&&(Pt.ambientLightColor.value=N.state.ambient,Pt.lightProbe.value=N.state.probe,Pt.directionalLights.value=N.state.directional,Pt.directionalLightShadows.value=N.state.directionalShadow,Pt.spotLights.value=N.state.spot,Pt.spotLightShadows.value=N.state.spotShadow,Pt.rectAreaLights.value=N.state.rectArea,Pt.ltc_1.value=N.state.rectAreaLTC1,Pt.ltc_2.value=N.state.rectAreaLTC2,Pt.pointLights.value=N.state.point,Pt.pointLightShadows.value=N.state.pointShadow,Pt.hemisphereLights.value=N.state.hemi,Pt.directionalShadowMap.value=N.state.directionalShadowMap,Pt.directionalShadowMatrix.value=N.state.directionalShadowMatrix,Pt.spotShadowMap.value=N.state.spotShadowMap,Pt.spotLightMatrix.value=N.state.spotLightMatrix,Pt.spotLightMap.value=N.state.spotLightMap,Pt.pointShadowMap.value=N.state.pointShadowMap,Pt.pointShadowMatrix.value=N.state.pointShadowMatrix),z.currentProgram=Nt,z.uniformsList=null,Nt}function pi(x){if(x.uniformsList===null){const U=x.currentProgram.getUniforms();x.uniformsList=_r.seqWithValue(U.seq,x.uniforms)}return x.uniformsList}function Ps(x,U){const O=Rt.get(x);O.outputColorSpace=U.outputColorSpace,O.batching=U.batching,O.batchingColor=U.batchingColor,O.instancing=U.instancing,O.instancingColor=U.instancingColor,O.instancingMorph=U.instancingMorph,O.skinning=U.skinning,O.morphTargets=U.morphTargets,O.morphNormals=U.morphNormals,O.morphColors=U.morphColors,O.morphTargetsCount=U.morphTargetsCount,O.numClippingPlanes=U.numClippingPlanes,O.numIntersection=U.numClipIntersection,O.vertexAlphas=U.vertexAlphas,O.vertexTangents=U.vertexTangents,O.toneMapping=U.toneMapping}function Ds(x,U,O,z,N){U.isScene!==!0&&(U=ie),b.resetTextureUnits();const et=U.fog,mt=z.isMeshStandardMaterial?U.environment:null,vt=I===null?y.outputColorSpace:I.isXRRenderTarget===!0?I.texture.colorSpace:Ji,Mt=(z.isMeshStandardMaterial?F:v).get(z.envMap||mt),It=z.vertexColors===!0&&!!O.attributes.color&&O.attributes.color.itemSize===4,Nt=!!O.attributes.tangent&&(!!z.normalMap||z.anisotropy>0),Pt=!!O.morphAttributes.position,Kt=!!O.morphAttributes.normal,te=!!O.morphAttributes.color;let xe=Wn;z.toneMapped&&(I===null||I.isXRRenderTarget===!0)&&(xe=y.toneMapping);const pe=O.morphAttributes.position||O.morphAttributes.normal||O.morphAttributes.color,Jt=pe!==void 0?pe.length:0,Dt=Rt.get(z),De=f.state.lights;if(nt===!0&&(_t===!0||x!==M)){const Be=x===M&&z.id===S;Z.setState(z,x,Be)}let ee=!1;z.version===Dt.__version?(Dt.needsLights&&Dt.lightsStateVersion!==De.state.version||Dt.outputColorSpace!==vt||N.isBatchedMesh&&Dt.batching===!1||!N.isBatchedMesh&&Dt.batching===!0||N.isBatchedMesh&&Dt.batchingColor===!0&&N.colorTexture===null||N.isBatchedMesh&&Dt.batchingColor===!1&&N.colorTexture!==null||N.isInstancedMesh&&Dt.instancing===!1||!N.isInstancedMesh&&Dt.instancing===!0||N.isSkinnedMesh&&Dt.skinning===!1||!N.isSkinnedMesh&&Dt.skinning===!0||N.isInstancedMesh&&Dt.instancingColor===!0&&N.instanceColor===null||N.isInstancedMesh&&Dt.instancingColor===!1&&N.instanceColor!==null||N.isInstancedMesh&&Dt.instancingMorph===!0&&N.morphTexture===null||N.isInstancedMesh&&Dt.instancingMorph===!1&&N.morphTexture!==null||Dt.envMap!==Mt||z.fog===!0&&Dt.fog!==et||Dt.numClippingPlanes!==void 0&&(Dt.numClippingPlanes!==Z.numPlanes||Dt.numIntersection!==Z.numIntersection)||Dt.vertexAlphas!==It||Dt.vertexTangents!==Nt||Dt.morphTargets!==Pt||Dt.morphNormals!==Kt||Dt.morphColors!==te||Dt.toneMapping!==xe||Dt.morphTargetsCount!==Jt)&&(ee=!0):(ee=!0,Dt.__version=z.version);let ln=Dt.currentProgram;ee===!0&&(ln=en(z,U,N));let mi=!1,qe=!1,is=!1;const de=ln.getUniforms(),nn=Dt.uniforms;if(At.useProgram(ln.program)&&(mi=!0,qe=!0,is=!0),z.id!==S&&(S=z.id,qe=!0),mi||M!==x){At.buffers.depth.getReversed()?(lt.copy(x.projectionMatrix),zh(lt),kh(lt),de.setValue(R,"projectionMatrix",lt)):de.setValue(R,"projectionMatrix",x.projectionMatrix),de.setValue(R,"viewMatrix",x.matrixWorldInverse);const Ge=de.map.cameraPosition;Ge!==void 0&&Ge.setValue(R,Lt.setFromMatrixPosition(x.matrixWorld)),jt.logarithmicDepthBuffer&&de.setValue(R,"logDepthBufFC",2/(Math.log(x.far+1)/Math.LN2)),(z.isMeshPhongMaterial||z.isMeshToonMaterial||z.isMeshLambertMaterial||z.isMeshBasicMaterial||z.isMeshStandardMaterial||z.isShaderMaterial)&&de.setValue(R,"isOrthographic",x.isOrthographicCamera===!0),M!==x&&(M=x,qe=!0,is=!0)}if(N.isSkinnedMesh){de.setOptional(R,N,"bindMatrix"),de.setOptional(R,N,"bindMatrixInverse");const Be=N.skeleton;Be&&(Be.boneTexture===null&&Be.computeBoneTexture(),de.setValue(R,"boneTexture",Be.boneTexture,b))}N.isBatchedMesh&&(de.setOptional(R,N,"batchingTexture"),de.setValue(R,"batchingTexture",N._matricesTexture,b),de.setOptional(R,N,"batchingIdTexture"),de.setValue(R,"batchingIdTexture",N._indirectTexture,b),de.setOptional(R,N,"batchingColorTexture"),N._colorsTexture!==null&&de.setValue(R,"batchingColorTexture",N._colorsTexture,b));const sn=O.morphAttributes;if((sn.position!==void 0||sn.normal!==void 0||sn.color!==void 0)&&wt.update(N,O,ln),(qe||Dt.receiveShadow!==N.receiveShadow)&&(Dt.receiveShadow=N.receiveShadow,de.setValue(R,"receiveShadow",N.receiveShadow)),z.isMeshGouraudMaterial&&z.envMap!==null&&(nn.envMap.value=Mt,nn.flipEnvMap.value=Mt.isCubeTexture&&Mt.isRenderTargetTexture===!1?-1:1),z.isMeshStandardMaterial&&z.envMap===null&&U.environment!==null&&(nn.envMapIntensity.value=U.environmentIntensity),qe&&(de.setValue(R,"toneMappingExposure",y.toneMappingExposure),Dt.needsLights&&Ir(nn,is),et&&z.fog===!0&&dt.refreshFogUniforms(nn,et),dt.refreshMaterialUniforms(nn,z,H,tt,f.state.transmissionRenderTarget[x.id]),_r.upload(R,pi(Dt),nn,b)),z.isShaderMaterial&&z.uniformsNeedUpdate===!0&&(_r.upload(R,pi(Dt),nn,b),z.uniformsNeedUpdate=!1),z.isSpriteMaterial&&de.setValue(R,"center",N.center),de.setValue(R,"modelViewMatrix",N.modelViewMatrix),de.setValue(R,"normalMatrix",N.normalMatrix),de.setValue(R,"modelMatrix",N.matrixWorld),z.isShaderMaterial||z.isRawShaderMaterial){const Be=z.uniformsGroups;for(let Ge=0,Or=Be.length;Ge<Or;Ge++){const jn=Be[Ge];L.update(jn,ln),L.bind(jn,ln)}}return ln}function Ir(x,U){x.ambientLightColor.needsUpdate=U,x.lightProbe.needsUpdate=U,x.directionalLights.needsUpdate=U,x.directionalLightShadows.needsUpdate=U,x.pointLights.needsUpdate=U,x.pointLightShadows.needsUpdate=U,x.spotLights.needsUpdate=U,x.spotLightShadows.needsUpdate=U,x.rectAreaLights.needsUpdate=U,x.hemisphereLights.needsUpdate=U}function Ls(x){return x.isMeshLambertMaterial||x.isMeshToonMaterial||x.isMeshPhongMaterial||x.isMeshStandardMaterial||x.isShadowMaterial||x.isShaderMaterial&&x.lights===!0}this.getActiveCubeFace=function(){return w},this.getActiveMipmapLevel=function(){return C},this.getRenderTarget=function(){return I},this.setRenderTargetTextures=function(x,U,O){Rt.get(x.texture).__webglTexture=U,Rt.get(x.depthTexture).__webglTexture=O;const z=Rt.get(x);z.__hasExternalTextures=!0,z.__autoAllocateDepthBuffer=O===void 0,z.__autoAllocateDepthBuffer||qt.has("WEBGL_multisampled_render_to_texture")===!0&&(console.warn("THREE.WebGLRenderer: Render-to-texture extension was disabled because an external texture was provided"),z.__useRenderToTexture=!1)},this.setRenderTargetFramebuffer=function(x,U){const O=Rt.get(x);O.__webglFramebuffer=U,O.__useDefaultFramebuffer=U===void 0};const Nr=R.createFramebuffer();this.setRenderTarget=function(x,U=0,O=0){I=x,w=U,C=O;let z=!0,N=null,et=!1,mt=!1;if(x){const Mt=Rt.get(x);if(Mt.__useDefaultFramebuffer!==void 0)At.bindFramebuffer(R.FRAMEBUFFER,null),z=!1;else if(Mt.__webglFramebuffer===void 0)b.setupRenderTarget(x);else if(Mt.__hasExternalTextures)b.rebindTextures(x,Rt.get(x.texture).__webglTexture,Rt.get(x.depthTexture).__webglTexture);else if(x.depthBuffer){const Pt=x.depthTexture;if(Mt.__boundDepthTexture!==Pt){if(Pt!==null&&Rt.has(Pt)&&(x.width!==Pt.image.width||x.height!==Pt.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");b.setupDepthRenderbuffer(x)}}const It=x.texture;(It.isData3DTexture||It.isDataArrayTexture||It.isCompressedArrayTexture)&&(mt=!0);const Nt=Rt.get(x).__webglFramebuffer;x.isWebGLCubeRenderTarget?(Array.isArray(Nt[U])?N=Nt[U][O]:N=Nt[U],et=!0):x.samples>0&&b.useMultisampledRTT(x)===!1?N=Rt.get(x).__webglMultisampledFramebuffer:Array.isArray(Nt)?N=Nt[O]:N=Nt,A.copy(x.viewport),W.copy(x.scissor),k=x.scissorTest}else A.copy(Et).multiplyScalar(H).floor(),W.copy(Ft).multiplyScalar(H).floor(),k=Xt;if(O!==0&&(N=Nr),At.bindFramebuffer(R.FRAMEBUFFER,N)&&z&&At.drawBuffers(x,N),At.viewport(A),At.scissor(W),At.setScissorTest(k),et){const Mt=Rt.get(x.texture);R.framebufferTexture2D(R.FRAMEBUFFER,R.COLOR_ATTACHMENT0,R.TEXTURE_CUBE_MAP_POSITIVE_X+U,Mt.__webglTexture,O)}else if(mt){const Mt=Rt.get(x.texture),It=U;R.framebufferTextureLayer(R.FRAMEBUFFER,R.COLOR_ATTACHMENT0,Mt.__webglTexture,O,It)}else if(x!==null&&O!==0){const Mt=Rt.get(x.texture);R.framebufferTexture2D(R.FRAMEBUFFER,R.COLOR_ATTACHMENT0,R.TEXTURE_2D,Mt.__webglTexture,O)}S=-1},this.readRenderTargetPixels=function(x,U,O,z,N,et,mt){if(!(x&&x.isWebGLRenderTarget)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let vt=Rt.get(x).__webglFramebuffer;if(x.isWebGLCubeRenderTarget&&mt!==void 0&&(vt=vt[mt]),vt){At.bindFramebuffer(R.FRAMEBUFFER,vt);try{const Mt=x.texture,It=Mt.format,Nt=Mt.type;if(!jt.textureFormatReadable(It)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!jt.textureTypeReadable(Nt)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}U>=0&&U<=x.width-z&&O>=0&&O<=x.height-N&&R.readPixels(U,O,z,N,Ut.convert(It),Ut.convert(Nt),et)}finally{const Mt=I!==null?Rt.get(I).__webglFramebuffer:null;At.bindFramebuffer(R.FRAMEBUFFER,Mt)}}},this.readRenderTargetPixelsAsync=async function(x,U,O,z,N,et,mt){if(!(x&&x.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let vt=Rt.get(x).__webglFramebuffer;if(x.isWebGLCubeRenderTarget&&mt!==void 0&&(vt=vt[mt]),vt){const Mt=x.texture,It=Mt.format,Nt=Mt.type;if(!jt.textureFormatReadable(It))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!jt.textureTypeReadable(Nt))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");if(U>=0&&U<=x.width-z&&O>=0&&O<=x.height-N){At.bindFramebuffer(R.FRAMEBUFFER,vt);const Pt=R.createBuffer();R.bindBuffer(R.PIXEL_PACK_BUFFER,Pt),R.bufferData(R.PIXEL_PACK_BUFFER,et.byteLength,R.STREAM_READ),R.readPixels(U,O,z,N,Ut.convert(It),Ut.convert(Nt),0);const Kt=I!==null?Rt.get(I).__webglFramebuffer:null;At.bindFramebuffer(R.FRAMEBUFFER,Kt);const te=R.fenceSync(R.SYNC_GPU_COMMANDS_COMPLETE,0);return R.flush(),await Bh(R,te,4),R.bindBuffer(R.PIXEL_PACK_BUFFER,Pt),R.getBufferSubData(R.PIXEL_PACK_BUFFER,0,et),R.deleteBuffer(Pt),R.deleteSync(te),et}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")}},this.copyFramebufferToTexture=function(x,U=null,O=0){x.isTexture!==!0&&(Fi("WebGLRenderer: copyFramebufferToTexture function signature has changed."),U=arguments[0]||null,x=arguments[1]);const z=Math.pow(2,-O),N=Math.floor(x.image.width*z),et=Math.floor(x.image.height*z),mt=U!==null?U.x:0,vt=U!==null?U.y:0;b.setTexture2D(x,0),R.copyTexSubImage2D(R.TEXTURE_2D,O,0,0,mt,vt,N,et),At.unbindTexture()};const Fr=R.createFramebuffer(),Oc=R.createFramebuffer();this.copyTextureToTexture=function(x,U,O=null,z=null,N=0,et=null){x.isTexture!==!0&&(Fi("WebGLRenderer: copyTextureToTexture function signature has changed."),z=arguments[0]||null,x=arguments[1],U=arguments[2],et=arguments[3]||0,O=null),et===null&&(N!==0?(Fi("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),et=N,N=0):et=0);let mt,vt,Mt,It,Nt,Pt,Kt,te,xe;const pe=x.isCompressedTexture?x.mipmaps[et]:x.image;if(O!==null)mt=O.max.x-O.min.x,vt=O.max.y-O.min.y,Mt=O.isBox3?O.max.z-O.min.z:1,It=O.min.x,Nt=O.min.y,Pt=O.isBox3?O.min.z:0;else{const sn=Math.pow(2,-N);mt=Math.floor(pe.width*sn),vt=Math.floor(pe.height*sn),x.isDataArrayTexture?Mt=pe.depth:x.isData3DTexture?Mt=Math.floor(pe.depth*sn):Mt=1,It=0,Nt=0,Pt=0}z!==null?(Kt=z.x,te=z.y,xe=z.z):(Kt=0,te=0,xe=0);const Jt=Ut.convert(U.format),Dt=Ut.convert(U.type);let De;U.isData3DTexture?(b.setTexture3D(U,0),De=R.TEXTURE_3D):U.isDataArrayTexture||U.isCompressedArrayTexture?(b.setTexture2DArray(U,0),De=R.TEXTURE_2D_ARRAY):(b.setTexture2D(U,0),De=R.TEXTURE_2D),R.pixelStorei(R.UNPACK_FLIP_Y_WEBGL,U.flipY),R.pixelStorei(R.UNPACK_PREMULTIPLY_ALPHA_WEBGL,U.premultiplyAlpha),R.pixelStorei(R.UNPACK_ALIGNMENT,U.unpackAlignment);const ee=R.getParameter(R.UNPACK_ROW_LENGTH),ln=R.getParameter(R.UNPACK_IMAGE_HEIGHT),mi=R.getParameter(R.UNPACK_SKIP_PIXELS),qe=R.getParameter(R.UNPACK_SKIP_ROWS),is=R.getParameter(R.UNPACK_SKIP_IMAGES);R.pixelStorei(R.UNPACK_ROW_LENGTH,pe.width),R.pixelStorei(R.UNPACK_IMAGE_HEIGHT,pe.height),R.pixelStorei(R.UNPACK_SKIP_PIXELS,It),R.pixelStorei(R.UNPACK_SKIP_ROWS,Nt),R.pixelStorei(R.UNPACK_SKIP_IMAGES,Pt);const de=x.isDataArrayTexture||x.isData3DTexture,nn=U.isDataArrayTexture||U.isData3DTexture;if(x.isDepthTexture){const sn=Rt.get(x),Be=Rt.get(U),Ge=Rt.get(sn.__renderTarget),Or=Rt.get(Be.__renderTarget);At.bindFramebuffer(R.READ_FRAMEBUFFER,Ge.__webglFramebuffer),At.bindFramebuffer(R.DRAW_FRAMEBUFFER,Or.__webglFramebuffer);for(let jn=0;jn<Mt;jn++)de&&(R.framebufferTextureLayer(R.READ_FRAMEBUFFER,R.COLOR_ATTACHMENT0,Rt.get(x).__webglTexture,N,Pt+jn),R.framebufferTextureLayer(R.DRAW_FRAMEBUFFER,R.COLOR_ATTACHMENT0,Rt.get(U).__webglTexture,et,xe+jn)),R.blitFramebuffer(It,Nt,mt,vt,Kt,te,mt,vt,R.DEPTH_BUFFER_BIT,R.NEAREST);At.bindFramebuffer(R.READ_FRAMEBUFFER,null),At.bindFramebuffer(R.DRAW_FRAMEBUFFER,null)}else if(N!==0||x.isRenderTargetTexture||Rt.has(x)){const sn=Rt.get(x),Be=Rt.get(U);At.bindFramebuffer(R.READ_FRAMEBUFFER,Fr),At.bindFramebuffer(R.DRAW_FRAMEBUFFER,Oc);for(let Ge=0;Ge<Mt;Ge++)de?R.framebufferTextureLayer(R.READ_FRAMEBUFFER,R.COLOR_ATTACHMENT0,sn.__webglTexture,N,Pt+Ge):R.framebufferTexture2D(R.READ_FRAMEBUFFER,R.COLOR_ATTACHMENT0,R.TEXTURE_2D,sn.__webglTexture,N),nn?R.framebufferTextureLayer(R.DRAW_FRAMEBUFFER,R.COLOR_ATTACHMENT0,Be.__webglTexture,et,xe+Ge):R.framebufferTexture2D(R.DRAW_FRAMEBUFFER,R.COLOR_ATTACHMENT0,R.TEXTURE_2D,Be.__webglTexture,et),N!==0?R.blitFramebuffer(It,Nt,mt,vt,Kt,te,mt,vt,R.COLOR_BUFFER_BIT,R.NEAREST):nn?R.copyTexSubImage3D(De,et,Kt,te,xe+Ge,It,Nt,mt,vt):R.copyTexSubImage2D(De,et,Kt,te,It,Nt,mt,vt);At.bindFramebuffer(R.READ_FRAMEBUFFER,null),At.bindFramebuffer(R.DRAW_FRAMEBUFFER,null)}else nn?x.isDataTexture||x.isData3DTexture?R.texSubImage3D(De,et,Kt,te,xe,mt,vt,Mt,Jt,Dt,pe.data):U.isCompressedArrayTexture?R.compressedTexSubImage3D(De,et,Kt,te,xe,mt,vt,Mt,Jt,pe.data):R.texSubImage3D(De,et,Kt,te,xe,mt,vt,Mt,Jt,Dt,pe):x.isDataTexture?R.texSubImage2D(R.TEXTURE_2D,et,Kt,te,mt,vt,Jt,Dt,pe.data):x.isCompressedTexture?R.compressedTexSubImage2D(R.TEXTURE_2D,et,Kt,te,pe.width,pe.height,Jt,pe.data):R.texSubImage2D(R.TEXTURE_2D,et,Kt,te,mt,vt,Jt,Dt,pe);R.pixelStorei(R.UNPACK_ROW_LENGTH,ee),R.pixelStorei(R.UNPACK_IMAGE_HEIGHT,ln),R.pixelStorei(R.UNPACK_SKIP_PIXELS,mi),R.pixelStorei(R.UNPACK_SKIP_ROWS,qe),R.pixelStorei(R.UNPACK_SKIP_IMAGES,is),et===0&&U.generateMipmaps&&R.generateMipmap(De),At.unbindTexture()},this.copyTextureToTexture3D=function(x,U,O=null,z=null,N=0){return x.isTexture!==!0&&(Fi("WebGLRenderer: copyTextureToTexture3D function signature has changed."),O=arguments[0]||null,z=arguments[1]||null,x=arguments[2],U=arguments[3],N=arguments[4]||0),Fi('WebGLRenderer: copyTextureToTexture3D function has been deprecated. Use "copyTextureToTexture" instead.'),this.copyTextureToTexture(x,U,O,z,N)},this.initRenderTarget=function(x){Rt.get(x).__webglFramebuffer===void 0&&b.setupRenderTarget(x)},this.initTexture=function(x){x.isCubeTexture?b.setTextureCube(x,0):x.isData3DTexture?b.setTexture3D(x,0):x.isDataArrayTexture||x.isCompressedArrayTexture?b.setTexture2DArray(x,0):b.setTexture2D(x,0),At.unbindTexture()},this.resetState=function(){w=0,C=0,I=null,At.reset(),Zt.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Rn}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(t){this._outputColorSpace=t;const e=this.getContext();e.drawingBufferColorspace=Qt._getDrawingBufferColorSpace(t),e.unpackColorSpace=Qt._getUnpackColorSpace()}}const Vl={type:"change"},Po={type:"start"},Uc={type:"end"},cr=new bs,Gl=new Hn,cg=Math.cos(70*Fh.DEG2RAD),Ee=new P,We=2*Math.PI,ae={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},_a=1e-6;class hg extends bu{constructor(t,e=null){super(t,e),this.state=ae.NONE,this.enabled=!0,this.target=new P,this.cursor=new P,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:Hi.ROTATE,MIDDLE:Hi.DOLLY,RIGHT:Hi.PAN},this.touches={ONE:Oi.ROTATE,TWO:Oi.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this._lastPosition=new P,this._lastQuaternion=new ci,this._lastTargetPosition=new P,this._quat=new ci().setFromUnitVectors(t.up,new P(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new gl,this._sphericalDelta=new gl,this._scale=1,this._panOffset=new P,this._rotateStart=new St,this._rotateEnd=new St,this._rotateDelta=new St,this._panStart=new St,this._panEnd=new St,this._panDelta=new St,this._dollyStart=new St,this._dollyEnd=new St,this._dollyDelta=new St,this._dollyDirection=new P,this._mouse=new St,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=dg.bind(this),this._onPointerDown=ug.bind(this),this._onPointerUp=fg.bind(this),this._onContextMenu=Mg.bind(this),this._onMouseWheel=gg.bind(this),this._onKeyDown=_g.bind(this),this._onTouchStart=vg.bind(this),this._onTouchMove=xg.bind(this),this._onMouseDown=pg.bind(this),this._onMouseMove=mg.bind(this),this._interceptControlDown=Sg.bind(this),this._interceptControlUp=yg.bind(this),this.domElement!==null&&this.connect(),this.update()}connect(){this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction="auto"}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(t){t.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=t}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(Vl),this.update(),this.state=ae.NONE}update(t=null){const e=this.object.position;Ee.copy(e).sub(this.target),Ee.applyQuaternion(this._quat),this._spherical.setFromVector3(Ee),this.autoRotate&&this.state===ae.NONE&&this._rotateLeft(this._getAutoRotationAngle(t)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let n=this.minAzimuthAngle,s=this.maxAzimuthAngle;isFinite(n)&&isFinite(s)&&(n<-Math.PI?n+=We:n>Math.PI&&(n-=We),s<-Math.PI?s+=We:s>Math.PI&&(s-=We),n<=s?this._spherical.theta=Math.max(n,Math.min(s,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(n+s)/2?Math.max(n,this._spherical.theta):Math.min(s,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let r=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{const a=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),r=a!=this._spherical.radius}if(Ee.setFromSpherical(this._spherical),Ee.applyQuaternion(this._quatInverse),e.copy(this.target).add(Ee),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let a=null;if(this.object.isPerspectiveCamera){const o=Ee.length();a=this._clampDistance(o*this._scale);const l=o-a;this.object.position.addScaledVector(this._dollyDirection,l),this.object.updateMatrixWorld(),r=!!l}else if(this.object.isOrthographicCamera){const o=new P(this._mouse.x,this._mouse.y,0);o.unproject(this.object);const l=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),r=l!==this.object.zoom;const c=new P(this._mouse.x,this._mouse.y,0);c.unproject(this.object),this.object.position.sub(c).add(o),this.object.updateMatrixWorld(),a=Ee.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;a!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(a).add(this.object.position):(cr.origin.copy(this.object.position),cr.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(cr.direction))<cg?this.object.lookAt(this.target):(Gl.setFromNormalAndCoplanarPoint(this.object.up,this.target),cr.intersectPlane(Gl,this.target))))}else if(this.object.isOrthographicCamera){const a=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),a!==this.object.zoom&&(this.object.updateProjectionMatrix(),r=!0)}return this._scale=1,this._performCursorZoom=!1,r||this._lastPosition.distanceToSquared(this.object.position)>_a||8*(1-this._lastQuaternion.dot(this.object.quaternion))>_a||this._lastTargetPosition.distanceToSquared(this.target)>_a?(this.dispatchEvent(Vl),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(t){return t!==null?We/60*this.autoRotateSpeed*t:We/60/60*this.autoRotateSpeed}_getZoomScale(t){const e=Math.abs(t*.01);return Math.pow(.95,this.zoomSpeed*e)}_rotateLeft(t){this._sphericalDelta.theta-=t}_rotateUp(t){this._sphericalDelta.phi-=t}_panLeft(t,e){Ee.setFromMatrixColumn(e,0),Ee.multiplyScalar(-t),this._panOffset.add(Ee)}_panUp(t,e){this.screenSpacePanning===!0?Ee.setFromMatrixColumn(e,1):(Ee.setFromMatrixColumn(e,0),Ee.crossVectors(this.object.up,Ee)),Ee.multiplyScalar(t),this._panOffset.add(Ee)}_pan(t,e){const n=this.domElement;if(this.object.isPerspectiveCamera){const s=this.object.position;Ee.copy(s).sub(this.target);let r=Ee.length();r*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*t*r/n.clientHeight,this.object.matrix),this._panUp(2*e*r/n.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(t*(this.object.right-this.object.left)/this.object.zoom/n.clientWidth,this.object.matrix),this._panUp(e*(this.object.top-this.object.bottom)/this.object.zoom/n.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(t){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=t:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(t){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=t:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(t,e){if(!this.zoomToCursor)return;this._performCursorZoom=!0;const n=this.domElement.getBoundingClientRect(),s=t-n.left,r=e-n.top,a=n.width,o=n.height;this._mouse.x=s/a*2-1,this._mouse.y=-(r/o)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(t){return Math.max(this.minDistance,Math.min(this.maxDistance,t))}_handleMouseDownRotate(t){this._rotateStart.set(t.clientX,t.clientY)}_handleMouseDownDolly(t){this._updateZoomParameters(t.clientX,t.clientX),this._dollyStart.set(t.clientX,t.clientY)}_handleMouseDownPan(t){this._panStart.set(t.clientX,t.clientY)}_handleMouseMoveRotate(t){this._rotateEnd.set(t.clientX,t.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const e=this.domElement;this._rotateLeft(We*this._rotateDelta.x/e.clientHeight),this._rotateUp(We*this._rotateDelta.y/e.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(t){this._dollyEnd.set(t.clientX,t.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(t){this._panEnd.set(t.clientX,t.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(t){this._updateZoomParameters(t.clientX,t.clientY),t.deltaY<0?this._dollyIn(this._getZoomScale(t.deltaY)):t.deltaY>0&&this._dollyOut(this._getZoomScale(t.deltaY)),this.update()}_handleKeyDown(t){let e=!1;switch(t.code){case this.keys.UP:t.ctrlKey||t.metaKey||t.shiftKey?this.enableRotate&&this._rotateUp(We*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),e=!0;break;case this.keys.BOTTOM:t.ctrlKey||t.metaKey||t.shiftKey?this.enableRotate&&this._rotateUp(-We*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),e=!0;break;case this.keys.LEFT:t.ctrlKey||t.metaKey||t.shiftKey?this.enableRotate&&this._rotateLeft(We*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),e=!0;break;case this.keys.RIGHT:t.ctrlKey||t.metaKey||t.shiftKey?this.enableRotate&&this._rotateLeft(-We*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),e=!0;break}e&&(t.preventDefault(),this.update())}_handleTouchStartRotate(t){if(this._pointers.length===1)this._rotateStart.set(t.pageX,t.pageY);else{const e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._rotateStart.set(n,s)}}_handleTouchStartPan(t){if(this._pointers.length===1)this._panStart.set(t.pageX,t.pageY);else{const e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._panStart.set(n,s)}}_handleTouchStartDolly(t){const e=this._getSecondPointerPosition(t),n=t.pageX-e.x,s=t.pageY-e.y,r=Math.sqrt(n*n+s*s);this._dollyStart.set(0,r)}_handleTouchStartDollyPan(t){this.enableZoom&&this._handleTouchStartDolly(t),this.enablePan&&this._handleTouchStartPan(t)}_handleTouchStartDollyRotate(t){this.enableZoom&&this._handleTouchStartDolly(t),this.enableRotate&&this._handleTouchStartRotate(t)}_handleTouchMoveRotate(t){if(this._pointers.length==1)this._rotateEnd.set(t.pageX,t.pageY);else{const n=this._getSecondPointerPosition(t),s=.5*(t.pageX+n.x),r=.5*(t.pageY+n.y);this._rotateEnd.set(s,r)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const e=this.domElement;this._rotateLeft(We*this._rotateDelta.x/e.clientHeight),this._rotateUp(We*this._rotateDelta.y/e.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(t){if(this._pointers.length===1)this._panEnd.set(t.pageX,t.pageY);else{const e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._panEnd.set(n,s)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(t){const e=this._getSecondPointerPosition(t),n=t.pageX-e.x,s=t.pageY-e.y,r=Math.sqrt(n*n+s*s);this._dollyEnd.set(0,r),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);const a=(t.pageX+e.x)*.5,o=(t.pageY+e.y)*.5;this._updateZoomParameters(a,o)}_handleTouchMoveDollyPan(t){this.enableZoom&&this._handleTouchMoveDolly(t),this.enablePan&&this._handleTouchMovePan(t)}_handleTouchMoveDollyRotate(t){this.enableZoom&&this._handleTouchMoveDolly(t),this.enableRotate&&this._handleTouchMoveRotate(t)}_addPointer(t){this._pointers.push(t.pointerId)}_removePointer(t){delete this._pointerPositions[t.pointerId];for(let e=0;e<this._pointers.length;e++)if(this._pointers[e]==t.pointerId){this._pointers.splice(e,1);return}}_isTrackingPointer(t){for(let e=0;e<this._pointers.length;e++)if(this._pointers[e]==t.pointerId)return!0;return!1}_trackPointer(t){let e=this._pointerPositions[t.pointerId];e===void 0&&(e=new St,this._pointerPositions[t.pointerId]=e),e.set(t.pageX,t.pageY)}_getSecondPointerPosition(t){const e=t.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[e]}_customWheelEvent(t){const e=t.deltaMode,n={clientX:t.clientX,clientY:t.clientY,deltaY:t.deltaY};switch(e){case 1:n.deltaY*=16;break;case 2:n.deltaY*=100;break}return t.ctrlKey&&!this._controlActive&&(n.deltaY*=10),n}}function ug(i){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(i.pointerId),this.domElement.addEventListener("pointermove",this._onPointerMove),this.domElement.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(i)&&(this._addPointer(i),i.pointerType==="touch"?this._onTouchStart(i):this._onMouseDown(i)))}function dg(i){this.enabled!==!1&&(i.pointerType==="touch"?this._onTouchMove(i):this._onMouseMove(i))}function fg(i){switch(this._removePointer(i),this._pointers.length){case 0:this.domElement.releasePointerCapture(i.pointerId),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(Uc),this.state=ae.NONE;break;case 1:const t=this._pointers[0],e=this._pointerPositions[t];this._onTouchStart({pointerId:t,pageX:e.x,pageY:e.y});break}}function pg(i){let t;switch(i.button){case 0:t=this.mouseButtons.LEFT;break;case 1:t=this.mouseButtons.MIDDLE;break;case 2:t=this.mouseButtons.RIGHT;break;default:t=-1}switch(t){case Hi.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(i),this.state=ae.DOLLY;break;case Hi.ROTATE:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=ae.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=ae.ROTATE}break;case Hi.PAN:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=ae.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=ae.PAN}break;default:this.state=ae.NONE}this.state!==ae.NONE&&this.dispatchEvent(Po)}function mg(i){switch(this.state){case ae.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(i);break;case ae.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(i);break;case ae.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(i);break}}function gg(i){this.enabled===!1||this.enableZoom===!1||this.state!==ae.NONE||(i.preventDefault(),this.dispatchEvent(Po),this._handleMouseWheel(this._customWheelEvent(i)),this.dispatchEvent(Uc))}function _g(i){this.enabled!==!1&&this._handleKeyDown(i)}function vg(i){switch(this._trackPointer(i),this._pointers.length){case 1:switch(this.touches.ONE){case Oi.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(i),this.state=ae.TOUCH_ROTATE;break;case Oi.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(i),this.state=ae.TOUCH_PAN;break;default:this.state=ae.NONE}break;case 2:switch(this.touches.TWO){case Oi.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(i),this.state=ae.TOUCH_DOLLY_PAN;break;case Oi.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(i),this.state=ae.TOUCH_DOLLY_ROTATE;break;default:this.state=ae.NONE}break;default:this.state=ae.NONE}this.state!==ae.NONE&&this.dispatchEvent(Po)}function xg(i){switch(this._trackPointer(i),this.state){case ae.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(i),this.update();break;case ae.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(i),this.update();break;case ae.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(i),this.update();break;case ae.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(i),this.update();break;default:this.state=ae.NONE}}function Mg(i){this.enabled!==!1&&i.preventDefault()}function Sg(i){i.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function yg(i){i.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}const Ic={name:"CopyShader",uniforms:{tDiffuse:{value:null},opacity:{value:1}},vertexShader:`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`

		uniform float opacity;

		uniform sampler2D tDiffuse;

		varying vec2 vUv;

		void main() {

			vec4 texel = texture2D( tDiffuse, vUv );
			gl_FragColor = opacity * texel;


		}`};class As{constructor(){this.isPass=!0,this.enabled=!0,this.needsSwap=!0,this.clear=!1,this.renderToScreen=!1}setSize(){}render(){console.error("THREE.Pass: .render() must be implemented in derived pass.")}dispose(){}}const Eg=new Ac(-1,1,1,-1,0,1);class bg extends ge{constructor(){super(),this.setAttribute("position",new Oe([-1,3,0,-1,-1,0,3,-1,0],3)),this.setAttribute("uv",new Oe([0,2,0,0,2,0],2))}}const Tg=new bg;class Nc{constructor(t){this._mesh=new be(Tg,t)}dispose(){this._mesh.geometry.dispose()}render(t){t.render(this._mesh,Eg)}get material(){return this._mesh.material}set material(t){this._mesh.material=t}}class vr extends As{constructor(t,e){super(),this.textureID=e!==void 0?e:"tDiffuse",t instanceof He?(this.uniforms=t.uniforms,this.material=t):t&&(this.uniforms=Tr.clone(t.uniforms),this.material=new He({name:t.name!==void 0?t.name:"unspecified",defines:Object.assign({},t.defines),uniforms:this.uniforms,vertexShader:t.vertexShader,fragmentShader:t.fragmentShader})),this.fsQuad=new Nc(this.material)}render(t,e,n){this.uniforms[this.textureID]&&(this.uniforms[this.textureID].value=n.texture),this.fsQuad.material=this.material,this.renderToScreen?(t.setRenderTarget(null),this.fsQuad.render(t)):(t.setRenderTarget(e),this.clear&&t.clear(t.autoClearColor,t.autoClearDepth,t.autoClearStencil),this.fsQuad.render(t))}dispose(){this.material.dispose(),this.fsQuad.dispose()}}class Wl extends As{constructor(t,e){super(),this.scene=t,this.camera=e,this.clear=!0,this.needsSwap=!1,this.inverse=!1}render(t,e,n){const s=t.getContext(),r=t.state;r.buffers.color.setMask(!1),r.buffers.depth.setMask(!1),r.buffers.color.setLocked(!0),r.buffers.depth.setLocked(!0);let a,o;this.inverse?(a=0,o=1):(a=1,o=0),r.buffers.stencil.setTest(!0),r.buffers.stencil.setOp(s.REPLACE,s.REPLACE,s.REPLACE),r.buffers.stencil.setFunc(s.ALWAYS,a,4294967295),r.buffers.stencil.setClear(o),r.buffers.stencil.setLocked(!0),t.setRenderTarget(n),this.clear&&t.clear(),t.render(this.scene,this.camera),t.setRenderTarget(e),this.clear&&t.clear(),t.render(this.scene,this.camera),r.buffers.color.setLocked(!1),r.buffers.depth.setLocked(!1),r.buffers.color.setMask(!0),r.buffers.depth.setMask(!0),r.buffers.stencil.setLocked(!1),r.buffers.stencil.setFunc(s.EQUAL,1,4294967295),r.buffers.stencil.setOp(s.KEEP,s.KEEP,s.KEEP),r.buffers.stencil.setLocked(!0)}}class wg extends As{constructor(){super(),this.needsSwap=!1}render(t){t.state.buffers.stencil.setLocked(!1),t.state.buffers.stencil.setTest(!1)}}class Ag{constructor(t,e){if(this.renderer=t,this._pixelRatio=t.getPixelRatio(),e===void 0){const n=t.getSize(new St);this._width=n.width,this._height=n.height,e=new fn(this._width*this._pixelRatio,this._height*this._pixelRatio,{type:Pn}),e.texture.name="EffectComposer.rt1"}else this._width=e.width,this._height=e.height;this.renderTarget1=e,this.renderTarget2=e.clone(),this.renderTarget2.texture.name="EffectComposer.rt2",this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2,this.renderToScreen=!0,this.passes=[],this.copyPass=new vr(Ic),this.copyPass.material.blending=Cn,this.clock=new yu}swapBuffers(){const t=this.readBuffer;this.readBuffer=this.writeBuffer,this.writeBuffer=t}addPass(t){this.passes.push(t),t.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}insertPass(t,e){this.passes.splice(e,0,t),t.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}removePass(t){const e=this.passes.indexOf(t);e!==-1&&this.passes.splice(e,1)}isLastEnabledPass(t){for(let e=t+1;e<this.passes.length;e++)if(this.passes[e].enabled)return!1;return!0}render(t){t===void 0&&(t=this.clock.getDelta());const e=this.renderer.getRenderTarget();let n=!1;for(let s=0,r=this.passes.length;s<r;s++){const a=this.passes[s];if(a.enabled!==!1){if(a.renderToScreen=this.renderToScreen&&this.isLastEnabledPass(s),a.render(this.renderer,this.writeBuffer,this.readBuffer,t,n),a.needsSwap){if(n){const o=this.renderer.getContext(),l=this.renderer.state.buffers.stencil;l.setFunc(o.NOTEQUAL,1,4294967295),this.copyPass.render(this.renderer,this.writeBuffer,this.readBuffer,t),l.setFunc(o.EQUAL,1,4294967295)}this.swapBuffers()}Wl!==void 0&&(a instanceof Wl?n=!0:a instanceof wg&&(n=!1))}}this.renderer.setRenderTarget(e)}reset(t){if(t===void 0){const e=this.renderer.getSize(new St);this._pixelRatio=this.renderer.getPixelRatio(),this._width=e.width,this._height=e.height,t=this.renderTarget1.clone(),t.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.renderTarget1=t,this.renderTarget2=t.clone(),this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2}setSize(t,e){this._width=t,this._height=e;const n=this._width*this._pixelRatio,s=this._height*this._pixelRatio;this.renderTarget1.setSize(n,s),this.renderTarget2.setSize(n,s);for(let r=0;r<this.passes.length;r++)this.passes[r].setSize(n,s)}setPixelRatio(t){this._pixelRatio=t,this.setSize(this._width,this._height)}dispose(){this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.copyPass.dispose()}}class Rg extends As{constructor(t,e,n=null,s=null,r=null){super(),this.scene=t,this.camera=e,this.overrideMaterial=n,this.clearColor=s,this.clearAlpha=r,this.clear=!0,this.clearDepth=!1,this.needsSwap=!1,this._oldClearColor=new ot}render(t,e,n){const s=t.autoClear;t.autoClear=!1;let r,a;this.overrideMaterial!==null&&(a=this.scene.overrideMaterial,this.scene.overrideMaterial=this.overrideMaterial),this.clearColor!==null&&(t.getClearColor(this._oldClearColor),t.setClearColor(this.clearColor,t.getClearAlpha())),this.clearAlpha!==null&&(r=t.getClearAlpha(),t.setClearAlpha(this.clearAlpha)),this.clearDepth==!0&&t.clearDepth(),t.setRenderTarget(this.renderToScreen?null:n),this.clear===!0&&t.clear(t.autoClearColor,t.autoClearDepth,t.autoClearStencil),t.render(this.scene,this.camera),this.clearColor!==null&&t.setClearColor(this._oldClearColor),this.clearAlpha!==null&&t.setClearAlpha(r),this.overrideMaterial!==null&&(this.scene.overrideMaterial=a),t.autoClear=s}}const Cg={uniforms:{tDiffuse:{value:null},luminosityThreshold:{value:1},smoothWidth:{value:1},defaultColor:{value:new ot(0)},defaultOpacity:{value:0}},vertexShader:`

		varying vec2 vUv;

		void main() {

			vUv = uv;

			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,fragmentShader:`

		uniform sampler2D tDiffuse;
		uniform vec3 defaultColor;
		uniform float defaultOpacity;
		uniform float luminosityThreshold;
		uniform float smoothWidth;

		varying vec2 vUv;

		void main() {

			vec4 texel = texture2D( tDiffuse, vUv );

			float v = luminance( texel.xyz );

			vec4 outputColor = vec4( defaultColor.rgb, defaultOpacity );

			float alpha = smoothstep( luminosityThreshold, luminosityThreshold + smoothWidth, v );

			gl_FragColor = mix( outputColor, texel, alpha );

		}`};class ts extends As{constructor(t,e,n,s){super(),this.strength=e!==void 0?e:1,this.radius=n,this.threshold=s,this.resolution=t!==void 0?new St(t.x,t.y):new St(256,256),this.clearColor=new ot(0,0,0),this.renderTargetsHorizontal=[],this.renderTargetsVertical=[],this.nMips=5;let r=Math.round(this.resolution.x/2),a=Math.round(this.resolution.y/2);this.renderTargetBright=new fn(r,a,{type:Pn}),this.renderTargetBright.texture.name="UnrealBloomPass.bright",this.renderTargetBright.texture.generateMipmaps=!1;for(let d=0;d<this.nMips;d++){const p=new fn(r,a,{type:Pn});p.texture.name="UnrealBloomPass.h"+d,p.texture.generateMipmaps=!1,this.renderTargetsHorizontal.push(p);const u=new fn(r,a,{type:Pn});u.texture.name="UnrealBloomPass.v"+d,u.texture.generateMipmaps=!1,this.renderTargetsVertical.push(u),r=Math.round(r/2),a=Math.round(a/2)}const o=Cg;this.highPassUniforms=Tr.clone(o.uniforms),this.highPassUniforms.luminosityThreshold.value=s,this.highPassUniforms.smoothWidth.value=.01,this.materialHighPassFilter=new He({uniforms:this.highPassUniforms,vertexShader:o.vertexShader,fragmentShader:o.fragmentShader}),this.separableBlurMaterials=[];const l=[3,5,7,9,11];r=Math.round(this.resolution.x/2),a=Math.round(this.resolution.y/2);for(let d=0;d<this.nMips;d++)this.separableBlurMaterials.push(this.getSeparableBlurMaterial(l[d])),this.separableBlurMaterials[d].uniforms.invSize.value=new St(1/r,1/a),r=Math.round(r/2),a=Math.round(a/2);this.compositeMaterial=this.getCompositeMaterial(this.nMips),this.compositeMaterial.uniforms.blurTexture1.value=this.renderTargetsVertical[0].texture,this.compositeMaterial.uniforms.blurTexture2.value=this.renderTargetsVertical[1].texture,this.compositeMaterial.uniforms.blurTexture3.value=this.renderTargetsVertical[2].texture,this.compositeMaterial.uniforms.blurTexture4.value=this.renderTargetsVertical[3].texture,this.compositeMaterial.uniforms.blurTexture5.value=this.renderTargetsVertical[4].texture,this.compositeMaterial.uniforms.bloomStrength.value=e,this.compositeMaterial.uniforms.bloomRadius.value=.1;const c=[1,.8,.6,.4,.2];this.compositeMaterial.uniforms.bloomFactors.value=c,this.bloomTintColors=[new P(1,1,1),new P(1,1,1),new P(1,1,1),new P(1,1,1),new P(1,1,1)],this.compositeMaterial.uniforms.bloomTintColors.value=this.bloomTintColors;const h=Ic;this.copyUniforms=Tr.clone(h.uniforms),this.blendMaterial=new He({uniforms:this.copyUniforms,vertexShader:h.vertexShader,fragmentShader:h.fragmentShader,blending:Le,depthTest:!1,depthWrite:!1,transparent:!0}),this.enabled=!0,this.needsSwap=!1,this._oldClearColor=new ot,this.oldClearAlpha=1,this.basic=new Ss,this.fsQuad=new Nc(null)}dispose(){for(let t=0;t<this.renderTargetsHorizontal.length;t++)this.renderTargetsHorizontal[t].dispose();for(let t=0;t<this.renderTargetsVertical.length;t++)this.renderTargetsVertical[t].dispose();this.renderTargetBright.dispose();for(let t=0;t<this.separableBlurMaterials.length;t++)this.separableBlurMaterials[t].dispose();this.compositeMaterial.dispose(),this.blendMaterial.dispose(),this.basic.dispose(),this.fsQuad.dispose()}setSize(t,e){let n=Math.round(t/2),s=Math.round(e/2);this.renderTargetBright.setSize(n,s);for(let r=0;r<this.nMips;r++)this.renderTargetsHorizontal[r].setSize(n,s),this.renderTargetsVertical[r].setSize(n,s),this.separableBlurMaterials[r].uniforms.invSize.value=new St(1/n,1/s),n=Math.round(n/2),s=Math.round(s/2)}render(t,e,n,s,r){t.getClearColor(this._oldClearColor),this.oldClearAlpha=t.getClearAlpha();const a=t.autoClear;t.autoClear=!1,t.setClearColor(this.clearColor,0),r&&t.state.buffers.stencil.setTest(!1),this.renderToScreen&&(this.fsQuad.material=this.basic,this.basic.map=n.texture,t.setRenderTarget(null),t.clear(),this.fsQuad.render(t)),this.highPassUniforms.tDiffuse.value=n.texture,this.highPassUniforms.luminosityThreshold.value=this.threshold,this.fsQuad.material=this.materialHighPassFilter,t.setRenderTarget(this.renderTargetBright),t.clear(),this.fsQuad.render(t);let o=this.renderTargetBright;for(let l=0;l<this.nMips;l++)this.fsQuad.material=this.separableBlurMaterials[l],this.separableBlurMaterials[l].uniforms.colorTexture.value=o.texture,this.separableBlurMaterials[l].uniforms.direction.value=ts.BlurDirectionX,t.setRenderTarget(this.renderTargetsHorizontal[l]),t.clear(),this.fsQuad.render(t),this.separableBlurMaterials[l].uniforms.colorTexture.value=this.renderTargetsHorizontal[l].texture,this.separableBlurMaterials[l].uniforms.direction.value=ts.BlurDirectionY,t.setRenderTarget(this.renderTargetsVertical[l]),t.clear(),this.fsQuad.render(t),o=this.renderTargetsVertical[l];this.fsQuad.material=this.compositeMaterial,this.compositeMaterial.uniforms.bloomStrength.value=this.strength,this.compositeMaterial.uniforms.bloomRadius.value=this.radius,this.compositeMaterial.uniforms.bloomTintColors.value=this.bloomTintColors,t.setRenderTarget(this.renderTargetsHorizontal[0]),t.clear(),this.fsQuad.render(t),this.fsQuad.material=this.blendMaterial,this.copyUniforms.tDiffuse.value=this.renderTargetsHorizontal[0].texture,r&&t.state.buffers.stencil.setTest(!0),this.renderToScreen?(t.setRenderTarget(null),this.fsQuad.render(t)):(t.setRenderTarget(n),this.fsQuad.render(t)),t.setClearColor(this._oldClearColor,this.oldClearAlpha),t.autoClear=a}getSeparableBlurMaterial(t){const e=[];for(let n=0;n<t;n++)e.push(.39894*Math.exp(-.5*n*n/(t*t))/t);return new He({defines:{KERNEL_RADIUS:t},uniforms:{colorTexture:{value:null},invSize:{value:new St(.5,.5)},direction:{value:new St(.5,.5)},gaussianCoefficients:{value:e}},vertexShader:`varying vec2 vUv;
				void main() {
					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,fragmentShader:`#include <common>
				varying vec2 vUv;
				uniform sampler2D colorTexture;
				uniform vec2 invSize;
				uniform vec2 direction;
				uniform float gaussianCoefficients[KERNEL_RADIUS];

				void main() {
					float weightSum = gaussianCoefficients[0];
					vec3 diffuseSum = texture2D( colorTexture, vUv ).rgb * weightSum;
					for( int i = 1; i < KERNEL_RADIUS; i ++ ) {
						float x = float(i);
						float w = gaussianCoefficients[i];
						vec2 uvOffset = direction * invSize * x;
						vec3 sample1 = texture2D( colorTexture, vUv + uvOffset ).rgb;
						vec3 sample2 = texture2D( colorTexture, vUv - uvOffset ).rgb;
						diffuseSum += (sample1 + sample2) * w;
						weightSum += 2.0 * w;
					}
					gl_FragColor = vec4(diffuseSum/weightSum, 1.0);
				}`})}getCompositeMaterial(t){return new He({defines:{NUM_MIPS:t},uniforms:{blurTexture1:{value:null},blurTexture2:{value:null},blurTexture3:{value:null},blurTexture4:{value:null},blurTexture5:{value:null},bloomStrength:{value:1},bloomFactors:{value:null},bloomTintColors:{value:null},bloomRadius:{value:0}},vertexShader:`varying vec2 vUv;
				void main() {
					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,fragmentShader:`varying vec2 vUv;
				uniform sampler2D blurTexture1;
				uniform sampler2D blurTexture2;
				uniform sampler2D blurTexture3;
				uniform sampler2D blurTexture4;
				uniform sampler2D blurTexture5;
				uniform float bloomStrength;
				uniform float bloomRadius;
				uniform float bloomFactors[NUM_MIPS];
				uniform vec3 bloomTintColors[NUM_MIPS];

				float lerpBloomFactor(const in float factor) {
					float mirrorFactor = 1.2 - factor;
					return mix(factor, mirrorFactor, bloomRadius);
				}

				void main() {
					gl_FragColor = bloomStrength * ( lerpBloomFactor(bloomFactors[0]) * vec4(bloomTintColors[0], 1.0) * texture2D(blurTexture1, vUv) +
						lerpBloomFactor(bloomFactors[1]) * vec4(bloomTintColors[1], 1.0) * texture2D(blurTexture2, vUv) +
						lerpBloomFactor(bloomFactors[2]) * vec4(bloomTintColors[2], 1.0) * texture2D(blurTexture3, vUv) +
						lerpBloomFactor(bloomFactors[3]) * vec4(bloomTintColors[3], 1.0) * texture2D(blurTexture4, vUv) +
						lerpBloomFactor(bloomFactors[4]) * vec4(bloomTintColors[4], 1.0) * texture2D(blurTexture5, vUv) );
				}`})}}ts.BlurDirectionX=new St(1,0);ts.BlurDirectionY=new St(0,1);function Pg(){const t=new Float32Array(6e3),e=new Float32Array(2e3*3);for(let r=0;r<2e3;r++){const a=Math.random()*Math.PI*2,o=Math.acos(2*Math.random()-1),l=600+Math.random()*400;t[r*3]=l*Math.sin(o)*Math.cos(a),t[r*3+1]=l*Math.sin(o)*Math.sin(a),t[r*3+2]=l*Math.cos(o);const c=Math.random();e[r*3]=.55+c*.25,e[r*3+1]=.55+c*.15,e[r*3+2]=.75+c*.25}const n=new ge;n.setAttribute("position",new he(t,3)),n.setAttribute("color",new he(e,3));const s=new oi({size:1.6,sizeAttenuation:!0,vertexColors:!0,transparent:!0,opacity:.6,depthWrite:!1,blending:Le});return new Yi(n,s)}function Dg(i){const t=new lu;t.background=new ot(328975),t.fog=new Dr(657946,.0035);const e=new $e(60,i.clientWidth/i.clientHeight,.1,2e3);e.position.set(0,30,80);const n=new lg({antialias:!0,alpha:!0,powerPreference:"high-performance"});n.setSize(i.clientWidth,i.clientHeight),n.setPixelRatio(Math.min(window.devicePixelRatio,2)),n.toneMapping=ec,n.toneMappingExposure=1.25,i.appendChild(n.domElement);const s=new hg(e,n.domElement);s.enableDamping=!0,s.dampingFactor=.05,s.rotateSpeed=.5,s.zoomSpeed=.8,s.minDistance=12,s.maxDistance=180,s.autoRotate=!0,s.autoRotateSpeed=.3;const r=new Ag(n);r.addPass(new Rg(t,e));const a=new ts(new St(i.clientWidth,i.clientHeight),.55,.6,.2);r.addPass(a);const o=new Mu(2763354,.7);t.add(o);const l=new dl(6514417,1.8,240);l.position.set(50,50,50),t.add(l);const c=new dl(11032055,1.2,240);c.position.set(-50,-30,-50),t.add(c);const h=Pg();t.add(h);const d=new Eu;d.params.Points={threshold:2};const p=new St;return{scene:t,camera:e,renderer:n,controls:s,composer:r,bloomPass:a,raycaster:d,mouse:p,lights:{ambient:o,point1:l,point2:c},starfield:h}}function Lg(i,t){const e=t.clientWidth,n=t.clientHeight;i.camera.aspect=e/n,i.camera.updateProjectionMatrix(),i.renderer.setSize(e,n),i.composer.setSize(e,n)}function Ug(i){i.scene.traverse(t=>{var e;(t instanceof be||t instanceof du)&&((e=t.geometry)==null||e.dispose(),Array.isArray(t.material)?t.material.forEach(n=>n.dispose()):t.material&&t.material.dispose())}),i.renderer.dispose(),i.composer.dispose()}class Ig{constructor(t){kt(this,"positions");kt(this,"velocities");kt(this,"running",!0);kt(this,"step",0);kt(this,"repulsionStrength",500);kt(this,"attractionStrength",.01);kt(this,"dampening",.9);kt(this,"baseMaxSteps",300);kt(this,"maxSteps",300);kt(this,"cooldownExtension",0);this.positions=t,this.velocities=new Map;for(const e of t.keys())this.velocities.set(e,new P)}addNode(t,e){this.positions.set(t,e.clone()),this.velocities.set(t,new P),this.cooldownExtension=100,this.maxSteps=Math.max(this.maxSteps,this.step+this.cooldownExtension),this.running=!0}removeNode(t){this.positions.delete(t),this.velocities.delete(t)}tick(t){if(!this.running)return;if(this.step>this.maxSteps){this.cooldownExtension>0&&(this.cooldownExtension=0,this.maxSteps=this.baseMaxSteps);return}this.step++;const e=Math.max(.001,1-this.step/this.maxSteps),n=Array.from(this.positions.keys());for(let s=0;s<n.length;s++)for(let r=s+1;r<n.length;r++){const a=this.positions.get(n[s]),o=this.positions.get(n[r]),l=new P().subVectors(a,o),c=l.length()||1,h=this.repulsionStrength/(c*c)*e,d=l.normalize().multiplyScalar(h);this.velocities.get(n[s]).add(d),this.velocities.get(n[r]).sub(d)}for(const s of t){const r=this.positions.get(s.source),a=this.positions.get(s.target);if(!r||!a)continue;const o=new P().subVectors(a,r),c=o.length()*this.attractionStrength*s.weight*e,h=o.normalize().multiplyScalar(c);this.velocities.get(s.source).add(h),this.velocities.get(s.target).sub(h)}for(const s of n){const r=this.positions.get(s),a=this.velocities.get(s);a.sub(r.clone().multiplyScalar(.001*e)),a.multiplyScalar(this.dampening),r.add(a)}}reset(){this.step=0,this.running=!0;for(const t of this.velocities.values())t.set(0,0,0)}}function Ng(i){return i>=.7?"active":i>=.4?"dormant":i>=.1?"silent":"unavailable"}const po={active:"#10b981",dormant:"#f59e0b",silent:"#8b5cf6",unavailable:"#6b7280"},Fg={active:"Easily retrievable (retention ≥ 70%)",dormant:"Retrievable with effort (40–70%)",silent:"Difficult, needs cues (10–40%)",unavailable:"Needs reinforcement (< 10%)"},xr={aha:"#FFD700",confusion:"#EF4444",failure:"#9CA3AF"},Og={aha:"Aha moments and breakthroughs",confusion:"Confusions and weak spots",failure:"Failures and guardrails"};function Xl(i,t){return t==="state"?po[Ng(i.retention)]:t==="ahagraph"?Bg(i)??Sa[i.type]??"#8B95A5":Sa[i.type]||"#8B95A5"}function Bg(i){const t=new Set((i.tags??[]).map(e=>e.toLowerCase()));return t.has("aha")?xr.aha:t.has("confusion")||t.has("weak-spot")?xr.confusion:t.has("failure")||t.has("guardrail")?xr.failure:null}let ms=null;function mo(){if(ms)return ms;const i=128,t=document.createElement("canvas");t.width=i,t.height=i;const e=t.getContext("2d");if(!e)return ms=new Pe,ms;const n=e.createRadialGradient(i/2,i/2,0,i/2,i/2,i/2);n.addColorStop(0,"rgba(255, 255, 255, 1.0)"),n.addColorStop(.25,"rgba(255, 255, 255, 0.7)"),n.addColorStop(.55,"rgba(255, 255, 255, 0.2)"),n.addColorStop(1,"rgba(255, 255, 255, 0.0)"),e.fillStyle=n,e.fillRect(0,0,i,i);const s=new bc(t);return s.needsUpdate=!0,ms=s,s}function Yl(i){if(i===0||i===1)return i;const t=.3;return Math.pow(2,-10*i)*Math.sin((i-t/4)*(2*Math.PI)/t)+1}function zg(i){return i*i*((1.70158+1)*i-1.70158)}class kg{constructor(){kt(this,"group");kt(this,"meshMap",new Map);kt(this,"glowMap",new Map);kt(this,"positions",new Map);kt(this,"labelSprites",new Map);kt(this,"hoveredNode",null);kt(this,"selectedNode",null);kt(this,"colorMode","type");kt(this,"materializingNodes",[]);kt(this,"dissolvingNodes",[]);kt(this,"growingNodes",[]);this.group=new zi}setColorMode(t){if(this.colorMode!==t){this.colorMode=t;for(const[e,n]of this.meshMap){const s=n.userData.retention??0,r=n.userData.type??"fact",a=Array.isArray(n.userData.tags)?n.userData.tags:[],l=Xl({type:r,retention:s,tags:a},t),c=new ot(l),h=n.material;h.color.copy(c),h.emissive.copy(c);const d=this.glowMap.get(e);d&&d.material.color.copy(c)}}}createNodes(t){const e=(1+Math.sqrt(5))/2,n=t.length;for(let s=0;s<n;s++){const r=t[s],a=1-2*s/(n-1||1),o=Math.sqrt(1-a*a),l=2*Math.PI*s/e,c=30+n*.5,h=new P(o*Math.cos(l)*c,a*c,o*Math.sin(l)*c);r.isCenter&&h.set(0,0,0),this.positions.set(r.id,h),this.createNodeMeshes(r,h,1)}return this.positions}createNodeMeshes(t,e,n){const s=.5+t.retention*2,r=Xl(t,this.colorMode),a=(t.suppression_count??0)>0,o=new Lr(s,16,16),l=new mu({color:new ot(r),emissive:new ot(r),emissiveIntensity:a?0:.3+t.retention*.5,roughness:.3,metalness:.1,transparent:!0,opacity:a?.2:.3+t.retention*.7}),c=new be(o,l);c.position.copy(e),c.scale.setScalar(n),c.userData={nodeId:t.id,type:t.type,retention:t.retention,tags:t.tags},this.meshMap.set(t.id,c),this.group.add(c);const h=new Xi({map:mo(),color:new ot(r),transparent:!0,opacity:n>0?a?.1:.3+t.retention*.35:0,blending:Le,depthWrite:!1}),d=new Bi(h);d.scale.set(s*6*n,s*6*n,1),d.position.copy(e),d.userData={isGlow:!0,nodeId:t.id},this.glowMap.set(t.id,d),this.group.add(d);const p=t.label||t.type,u=this.createTextSprite(p,"#94a3b8");return u.position.copy(e),u.position.y+=s*2+1.5,u.userData={isLabel:!0,nodeId:t.id,offset:s*2+1.5},this.group.add(u),this.labelSprites.set(t.id,u),{mesh:c,glow:d,label:u,size:s}}addNode(t,e,n={}){const s=(e==null?void 0:e.clone())??new P((Math.random()-.5)*40,(Math.random()-.5)*40,(Math.random()-.5)*40);this.positions.set(t.id,s);const{mesh:r,glow:a,label:o}=this.createNodeMeshes(t,s,0);return r.scale.setScalar(.001),a.scale.set(.001,.001,1),a.material.opacity=0,o.material.opacity=0,n.isBirthRitual?(r.visible=!1,a.visible=!1,o.visible=!1,r.userData.birthRitualPending={totalFrames:30,targetScale:.5+t.retention*2}):this.materializingNodes.push({id:t.id,frame:0,totalFrames:30,mesh:r,glow:a,label:o,targetScale:.5+t.retention*2}),s}igniteNode(t){const e=this.meshMap.get(t),n=this.glowMap.get(t),s=this.labelSprites.get(t);if(!e||!n||!s)return;const r=e.userData.birthRitualPending;r&&(e.visible=!0,n.visible=!0,s.visible=!0,delete e.userData.birthRitualPending,this.materializingNodes.push({id:t,frame:0,totalFrames:r.totalFrames,mesh:e,glow:n,label:s,targetScale:r.targetScale}))}removeNode(t){const e=this.meshMap.get(t),n=this.glowMap.get(t),s=this.labelSprites.get(t);!e||!n||!s||(this.materializingNodes=this.materializingNodes.filter(r=>r.id!==t),this.dissolvingNodes.push({id:t,frame:0,totalFrames:60,mesh:e,glow:n,label:s,originalScale:e.scale.x}))}growNode(t,e){const n=this.meshMap.get(t);if(!n)return;const s=n.scale.x,r=.5+e*2;n.userData.retention=e,this.growingNodes.push({id:t,frame:0,totalFrames:30,startScale:s,targetScale:r})}createTextSprite(t,e){const n=document.createElement("canvas"),s=n.getContext("2d");if(!s){const f=new Pe;return new Bi(new Xi({map:f,transparent:!0,opacity:0}))}n.width=512,n.height=64;const r=t.length>40?t.slice(0,37)+"...":t;s.clearRect(0,0,n.width,n.height),s.font='600 22px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';const o=s.measureText(r).width,c=Math.min(o+14*2,n.width-4),h=40,d=(n.width-c)/2,p=(n.height-h)/2,u=h/2;s.fillStyle="rgba(10, 16, 28, 0.82)",s.beginPath(),s.moveTo(d+u,p),s.lineTo(d+c-u,p),s.quadraticCurveTo(d+c,p,d+c,p+u),s.lineTo(d+c,p+h-u),s.quadraticCurveTo(d+c,p+h,d+c-u,p+h),s.lineTo(d+u,p+h),s.quadraticCurveTo(d,p+h,d,p+h-u),s.lineTo(d,p+u),s.quadraticCurveTo(d,p,d+u,p),s.closePath(),s.fill(),s.strokeStyle="rgba(148, 163, 184, 0.18)",s.lineWidth=1,s.stroke(),s.textAlign="center",s.textBaseline="middle",s.fillStyle=e,s.fillText(r,n.width/2,n.height/2+1);const g=new bc(n);g.needsUpdate=!0;const _=new Xi({map:g,transparent:!0,opacity:0,depthTest:!1,sizeAttenuation:!0}),m=new Bi(_);return m.scale.set(9,1.2,1),m}updatePositions(){this.group.children.forEach(t=>{if(t.userData.nodeId){const e=this.positions.get(t.userData.nodeId);if(!e)return;t.userData.isGlow?t.position.copy(e):t.userData.isLabel?(t.position.copy(e),t.position.y+=t.userData.offset):t instanceof be&&t.position.copy(e)}})}animate(t,e,n,s=1){var a,o;for(let l=this.materializingNodes.length-1;l>=0;l--){const c=this.materializingNodes[l];c.frame++;const h=Math.min(c.frame/c.totalFrames,1),d=Yl(h);if(c.mesh.scale.setScalar(Math.max(.001,d)),c.frame>=5){const p=Math.min((c.frame-5)/5,1),u=c.glow.material;u.opacity=p*.4;const g=c.targetScale*6*d;c.glow.scale.set(g,g,1)}if(c.frame>=40){const p=Math.min((c.frame-40)/20,1);c.label.material.opacity=p*.9}c.frame>=60&&this.materializingNodes.splice(l,1)}for(let l=this.dissolvingNodes.length-1;l>=0;l--){const c=this.dissolvingNodes[l];c.frame++;const h=Math.min(c.frame/c.totalFrames,1),d=1-zg(h),p=Math.max(.001,c.originalScale*d);c.mesh.scale.setScalar(p);const u=p*6;c.glow.scale.set(u,u,1);const g=c.mesh.material;g.opacity*=.97,c.glow.material.opacity*=.95,c.label.material.opacity*=.93,c.frame>=c.totalFrames&&(this.group.remove(c.mesh),this.group.remove(c.glow),this.group.remove(c.label),c.mesh.geometry.dispose(),c.mesh.material.dispose(),(a=c.glow.material.map)==null||a.dispose(),c.glow.material.dispose(),(o=c.label.material.map)==null||o.dispose(),c.label.material.dispose(),this.meshMap.delete(c.id),this.glowMap.delete(c.id),this.labelSprites.delete(c.id),this.positions.delete(c.id),this.dissolvingNodes.splice(l,1))}for(let l=this.growingNodes.length-1;l>=0;l--){const c=this.growingNodes[l];c.frame++;const h=Math.min(c.frame/c.totalFrames,1),d=c.startScale+(c.targetScale-c.startScale)*Yl(h),p=this.meshMap.get(c.id);p&&p.scale.setScalar(d);const u=this.glowMap.get(c.id);if(u){const g=d*6;u.scale.set(g,g,1)}c.frame>=c.totalFrames&&this.growingNodes.splice(l,1)}const r=new Set([...this.materializingNodes.map(l=>l.id),...this.dissolvingNodes.map(l=>l.id),...this.growingNodes.map(l=>l.id)]);this.meshMap.forEach((l,c)=>{if(r.has(c))return;const h=e.find(T=>T.id===c);if(!h)return;const d=1+Math.sin(t*1.5+e.indexOf(h)*.5)*.15*h.retention;l.scale.setScalar(d);const p=this.positions.get(c),u=p?n.position.distanceTo(p):0,g=1+Math.min(1.4,Math.max(0,(u-60)/100)),_=l.material;if(c===this.hoveredNode)_.emissiveIntensity=1*s;else if(c===this.selectedNode)_.emissiveIntensity=.8*s;else{const E=.3+h.retention*.5+Math.sin(t*(.8+h.retention*.7))*.1*h.retention;_.emissiveIntensity=E*s*g}const m=.3+h.retention*.7;_.opacity=Math.min(1,m*s*g);const f=this.glowMap.get(c);if(f){const T=f.material,E=.3+h.retention*.35;T.opacity=Math.min(.95,E*s*g)}}),this.labelSprites.forEach((l,c)=>{if(r.has(c))return;const h=this.positions.get(c);if(!h)return;const d=n.position.distanceTo(h),p=l.material,u=c===this.hoveredNode||c===this.selectedNode?1:d<40?.9:d<80?.9*(1-(d-40)/40):0;p.opacity+=(u-p.opacity)*.1})}getMeshes(){return Array.from(this.meshMap.values())}dispose(){this.group.traverse(t=>{var e,n,s,r,a;t instanceof be?((e=t.geometry)==null||e.dispose(),(n=t.material)==null||n.dispose()):t instanceof Bi&&((r=(s=t.material)==null?void 0:s.map)==null||r.dispose(),(a=t.material)==null||a.dispose())}),this.materializingNodes=[],this.dissolvingNodes=[],this.growingNodes=[]}}function Hg(i){return 1-Math.pow(1-i,3)}class Vg{constructor(){kt(this,"group");kt(this,"growingEdges",[]);kt(this,"dissolvingEdges",[]);this.group=new zi}createEdges(t,e){for(const n of t){const s=e.get(n.source),r=e.get(n.target);if(!s||!r)continue;const a=[s,r],o=new ge().setFromPoints(a),l=new Ar({color:9133302,transparent:!0,opacity:Math.min(.25+n.weight*.5,.8),blending:Le,depthWrite:!1}),c=new co(o,l);c.userData={source:n.source,target:n.target},this.group.add(c)}}addEdge(t,e){const n=e.get(t.source),s=e.get(t.target);if(!n||!s)return;const r=[n.clone(),n.clone()],a=new ge().setFromPoints(r),o=new Ar({color:9133302,transparent:!0,opacity:0,blending:Le,depthWrite:!1}),l=new co(a,o);l.userData={source:t.source,target:t.target},this.group.add(l),this.growingEdges.push({line:l,source:t.source,target:t.target,frame:0,totalFrames:45})}removeEdgesForNode(t){const e=[];this.group.children.forEach(n=>{const s=n;(s.userData.source===t||s.userData.target===t)&&e.push(s)});for(const n of e)this.growingEdges=this.growingEdges.filter(s=>s.line!==n),this.dissolvingEdges.push({line:n,frame:0,totalFrames:40})}animateEdges(t){for(let e=this.growingEdges.length-1;e>=0;e--){const n=this.growingEdges[e];n.frame++;const s=Hg(Math.min(n.frame/n.totalFrames,1)),r=t.get(n.source),a=t.get(n.target);if(!r||!a)continue;const o=r.clone().lerp(a,s),l=n.line.geometry.attributes.position;l.setXYZ(0,r.x,r.y,r.z),l.setXYZ(1,o.x,o.y,o.z),l.needsUpdate=!0;const c=n.line.material;c.opacity=s*.65,n.frame>=n.totalFrames&&(c.opacity=.65,this.growingEdges.splice(e,1))}for(let e=this.dissolvingEdges.length-1;e>=0;e--){const n=this.dissolvingEdges[e];n.frame++;const s=n.frame/n.totalFrames,r=n.line.material;r.opacity=Math.max(0,.65*(1-s)),n.frame>=n.totalFrames&&(this.group.remove(n.line),n.line.geometry.dispose(),n.line.material.dispose(),this.dissolvingEdges.splice(e,1))}}updatePositions(t){this.group.children.forEach(e=>{const n=e;if(this.growingEdges.some(a=>a.line===n)||this.dissolvingEdges.some(a=>a.line===n))return;const s=t.get(n.userData.source),r=t.get(n.userData.target);if(s&&r){const a=n.geometry.attributes.position;a.setXYZ(0,s.x,s.y,s.z),a.setXYZ(1,r.x,r.y,r.z),a.needsUpdate=!0}})}dispose(){this.group.children.forEach(t=>{var n,s;const e=t;(n=e.geometry)==null||n.dispose(),(s=e.material)==null||s.dispose()}),this.growingEdges=[],this.dissolvingEdges=[]}}class Gg{constructor(t){kt(this,"starField");kt(this,"neuralParticles");this.starField=this.createStarField(),this.neuralParticles=this.createNeuralParticles(),t.add(this.starField),t.add(this.neuralParticles)}createStarField(){const e=new ge,n=new Float32Array(3e3*3),s=new Float32Array(3e3);for(let a=0;a<3e3;a++)n[a*3]=(Math.random()-.5)*1e3,n[a*3+1]=(Math.random()-.5)*1e3,n[a*3+2]=(Math.random()-.5)*1e3,s[a]=Math.random()*1.5;e.setAttribute("position",new he(n,3)),e.setAttribute("size",new he(s,1));const r=new oi({color:6514417,size:.5,transparent:!0,opacity:.4,sizeAttenuation:!0,blending:Le});return new Yi(e,r)}createNeuralParticles(){const e=new ge,n=new Float32Array(500*3),s=new Float32Array(500*3);for(let a=0;a<500;a++)n[a*3]=(Math.random()-.5)*100,n[a*3+1]=(Math.random()-.5)*100,n[a*3+2]=(Math.random()-.5)*100,s[a*3]=.4+Math.random()*.3,s[a*3+1]=.3+Math.random()*.2,s[a*3+2]=.8+Math.random()*.2;e.setAttribute("position",new he(n,3)),e.setAttribute("color",new he(s,3));const r=new oi({size:.3,vertexColors:!0,transparent:!0,opacity:.4,blending:Le,sizeAttenuation:!0});return new Yi(e,r)}animate(t){this.starField.rotation.y+=1e-4,this.starField.rotation.x+=5e-5;const e=this.neuralParticles.geometry.attributes.position;for(let n=0;n<e.count;n++)e.setY(n,e.getY(n)+Math.sin(t+n*.1)*.02),e.setX(n,e.getX(n)+Math.cos(t+n*.05)*.01);e.needsUpdate=!0}dispose(){this.starField.geometry.dispose(),this.starField.material.dispose(),this.neuralParticles.geometry.dispose(),this.neuralParticles.material.dispose()}}class Wg{constructor(t){kt(this,"pulseEffects",[]);kt(this,"spawnBursts",[]);kt(this,"rainbowBursts",[]);kt(this,"rippleWaves",[]);kt(this,"implosions",[]);kt(this,"shockwaves",[]);kt(this,"connectionFlashes",[]);kt(this,"birthOrbs",[]);kt(this,"scene");this.scene=t}addPulse(t,e,n,s){this.pulseEffects.push({nodeId:t,intensity:e,color:n,decay:s})}createSpawnBurst(t,e){const s=new ge,r=new Float32Array(180),a=new Float32Array(180);for(let c=0;c<60;c++){r[c*3]=t.x,r[c*3+1]=t.y,r[c*3+2]=t.z;const h=Math.random()*Math.PI*2,d=Math.acos(2*Math.random()-1),p=.3+Math.random()*.5;a[c*3]=Math.sin(d)*Math.cos(h)*p,a[c*3+1]=Math.sin(d)*Math.sin(h)*p,a[c*3+2]=Math.cos(d)*p}s.setAttribute("position",new he(r,3)),s.setAttribute("velocity",new he(a,3));const o=new oi({color:e,size:.6,transparent:!0,opacity:1,blending:Le,sizeAttenuation:!0}),l=new Yi(s,o);this.scene.add(l),this.spawnBursts.push({position:t.clone(),age:0,particles:l})}createShockwave(t,e,n){const s=new Ro(.1,.5,64),r=new Ss({color:e,transparent:!0,opacity:.8,side:gn,blending:Le}),a=new be(s,r);a.position.copy(t),a.lookAt(n.position),this.scene.add(a),this.shockwaves.push({mesh:a,age:0,maxAge:60})}createRainbowBurst(t,e){const s=new ge,r=new Float32Array(360),a=new Float32Array(360),o=new Float32Array(120);for(let h=0;h<120;h++){r[h*3]=t.x,r[h*3+1]=t.y,r[h*3+2]=t.z;const d=Math.random()*Math.PI*2,p=Math.acos(2*Math.random()-1),u=.2+Math.random()*.6;a[h*3]=Math.sin(p)*Math.cos(d)*u,a[h*3+1]=Math.sin(p)*Math.sin(d)*u,a[h*3+2]=Math.cos(p)*u,o[h]=Math.random()}s.setAttribute("position",new he(r,3)),s.setAttribute("velocity",new he(a,3)),s.setAttribute("hueOffset",new he(o,1));const l=new oi({color:e,size:.8,transparent:!0,opacity:1,blending:Le,sizeAttenuation:!0}),c=new Yi(s,l);this.scene.add(c),this.rainbowBursts.push({position:t.clone(),age:0,maxAge:180,particles:c,baseColor:e.clone()})}createRippleWave(t){this.rippleWaves.push({origin:t.clone(),radius:0,speed:1.2,age:0,maxAge:90,pulsedNodes:new Set})}createImplosion(t,e){const s=new ge,r=new Float32Array(120),a=new Float32Array(120),o=8;for(let h=0;h<40;h++){const d=Math.random()*Math.PI*2,p=Math.acos(2*Math.random()-1),u=o*(.5+Math.random()*.5);r[h*3]=t.x+Math.sin(p)*Math.cos(d)*u,r[h*3+1]=t.y+Math.sin(p)*Math.sin(d)*u,r[h*3+2]=t.z+Math.cos(p)*u,a[h*3]=(t.x-r[h*3])*.04,a[h*3+1]=(t.y-r[h*3+1])*.04,a[h*3+2]=(t.z-r[h*3+2])*.04}s.setAttribute("position",new he(r,3)),s.setAttribute("velocity",new he(a,3));const l=new oi({color:e,size:.5,transparent:!0,opacity:1,blending:Le,sizeAttenuation:!0}),c=new Yi(s,l);this.scene.add(c),this.implosions.push({position:t.clone(),age:0,maxAge:60,particles:c,flash:null})}createConnectionFlash(t,e,n){const s=[t.clone(),e.clone()],r=new ge().setFromPoints(s),a=new Ar({color:n,transparent:!0,opacity:1,blending:Le}),o=new co(r,a);this.scene.add(o),this.connectionFlashes.push({line:o,intensity:1})}createBirthOrb(t,e,n,s,r={}){var _;const a=r.gestationFrames??48,o=r.flightFrames??90,l=r.distanceFromCamera??40,c=new P(0,0,-l).applyQuaternion(t.quaternion).add(t.position),h=new Xi({map:mo(),color:e.clone(),transparent:!0,opacity:0,blending:Le,depthWrite:!1,depthTest:!1}),d=new Bi(h);d.position.copy(c),d.scale.set(.5,.5,1),d.renderOrder=999;const p=new Xi({map:mo(),color:new ot(16777215),transparent:!0,opacity:0,blending:Le,depthWrite:!1,depthTest:!1}),u=new Bi(p);u.position.copy(c),u.scale.set(.2,.2,1),u.renderOrder=1e3,this.scene.add(d),this.scene.add(u);const g=((_=n())==null?void 0:_.clone())??c.clone();this.birthOrbs.push({sprite:d,core:u,startPos:c,getTargetPos:n,color:e.clone(),age:0,gestationFrames:a,flightFrames:o,arriveFired:!1,onArrive:s,lastTargetPos:g,aborted:!1})}update(t,e,n){for(let s=this.pulseEffects.length-1;s>=0;s--){const r=this.pulseEffects[s];if(r.intensity-=r.decay,r.intensity<=0){this.pulseEffects.splice(s,1);continue}const a=t.get(r.nodeId);if(a){const o=a.material;o.emissive.lerp(r.color,r.intensity*.3),o.emissiveIntensity=Math.max(o.emissiveIntensity,r.intensity)}}for(let s=this.spawnBursts.length-1;s>=0;s--){const r=this.spawnBursts[s];if(r.age++,r.age>120){this.scene.remove(r.particles),r.particles.geometry.dispose(),r.particles.material.dispose(),this.spawnBursts.splice(s,1);continue}const a=r.particles.geometry.attributes.position,o=r.particles.geometry.attributes.velocity;for(let c=0;c<a.count;c++)a.setX(c,a.getX(c)+o.getX(c)),a.setY(c,a.getY(c)+o.getY(c)),a.setZ(c,a.getZ(c)+o.getZ(c)),o.setX(c,o.getX(c)*.97),o.setY(c,o.getY(c)*.97),o.setZ(c,o.getZ(c)*.97);a.needsUpdate=!0;const l=r.particles.material;l.opacity=Math.max(0,1-r.age/120),l.size=.6*(1-r.age/200)}for(let s=this.rainbowBursts.length-1;s>=0;s--){const r=this.rainbowBursts[s];if(r.age++,r.age>r.maxAge){this.scene.remove(r.particles),r.particles.geometry.dispose(),r.particles.material.dispose(),this.rainbowBursts.splice(s,1);continue}const a=r.particles.geometry.attributes.position,o=r.particles.geometry.attributes.velocity;for(let p=0;p<a.count;p++)a.setX(p,a.getX(p)+o.getX(p)),a.setY(p,a.getY(p)+o.getY(p)),a.setZ(p,a.getZ(p)+o.getZ(p)),o.setX(p,o.getX(p)*.98),o.setY(p,o.getY(p)*.98),o.setZ(p,o.getZ(p)*.98);a.needsUpdate=!0;const l=r.age/r.maxAge,c=r.particles.material,h=r.age*.02%1,d=new ot().setHSL(h,1,.6);c.color.copy(r.baseColor).lerp(d,.6),c.opacity=Math.max(0,1-l*l),c.size=.8*(1-l*.5)*(1+Math.sin(r.age*.3)*.2)}if(n)for(let s=this.rippleWaves.length-1;s>=0;s--){const r=this.rippleWaves[s];if(r.age++,r.radius+=r.speed,r.age>r.maxAge){this.rippleWaves.splice(s,1);continue}const a=r.radius,o=3;n.forEach((l,c)=>{if(r.pulsedNodes.has(c))return;const h=l.distanceTo(r.origin);h>=a-o&&h<=a+o&&(r.pulsedNodes.add(c),this.addPulse(c,.8,new ot(65489),.03))})}for(let s=this.implosions.length-1;s>=0;s--){const r=this.implosions[s];if(r.age++,r.age>r.maxAge+20){this.scene.remove(r.particles),r.particles.geometry.dispose(),r.particles.material.dispose(),r.flash&&(this.scene.remove(r.flash),r.flash.geometry.dispose(),r.flash.material.dispose()),this.implosions.splice(s,1);continue}if(r.age<=r.maxAge){const a=r.particles.geometry.attributes.position,o=r.particles.geometry.attributes.velocity,l=1+r.age*.02;for(let h=0;h<a.count;h++)a.setX(h,a.getX(h)+o.getX(h)*l),a.setY(h,a.getY(h)+o.getY(h)*l),a.setZ(h,a.getZ(h)+o.getZ(h)*l);a.needsUpdate=!0;const c=r.particles.material;c.opacity=Math.min(1,r.age/15),c.size=.5+r.age/r.maxAge*.3}if(r.age===r.maxAge&&!r.flash){const a=new Lr(2,16,16),o=new Ss({color:16777215,transparent:!0,opacity:1,blending:Le});r.flash=new be(a,o),r.flash.position.copy(r.position),this.scene.add(r.flash),r.particles.material.opacity=0}if(r.flash&&r.age>r.maxAge){const a=(r.age-r.maxAge)/20;r.flash.material.opacity=Math.max(0,1-a),r.flash.scale.setScalar(1+a*3)}}for(let s=this.shockwaves.length-1;s>=0;s--){const r=this.shockwaves[s];if(r.age++,r.age>r.maxAge){this.scene.remove(r.mesh),r.mesh.geometry.dispose(),r.mesh.material.dispose(),this.shockwaves.splice(s,1);continue}const a=r.age/r.maxAge;r.mesh.scale.setScalar(1+a*20),r.mesh.material.opacity=.8*(1-a),r.mesh.lookAt(e.position)}for(let s=this.connectionFlashes.length-1;s>=0;s--){const r=this.connectionFlashes[s];if(r.intensity-=.015,r.intensity<=0){this.scene.remove(r.line),r.line.geometry.dispose(),r.line.material.dispose(),this.connectionFlashes.splice(s,1);continue}r.line.material.opacity=r.intensity}for(let s=this.birthOrbs.length-1;s>=0;s--){const r=this.birthOrbs[s];r.age++;const a=r.gestationFrames+r.flightFrames,o=r.sprite.material,l=r.core.material,c=r.getTargetPos();if(c)r.lastTargetPos.copy(c);else if(r.age>r.gestationFrames&&!r.aborted){r.aborted=!0;const h=r.sprite.position;o.color.setRGB(1,.15,.2),l.color.setRGB(1,.6,.6),this.createImplosion(h,new ot(16721203)),r.arriveFired=!0,r.age=a+1}if(r.age<=r.gestationFrames){const h=r.age/r.gestationFrames,d=1-Math.pow(1-h,3),p=.85+Math.sin(r.age*.35)*.15,u=.5+d*4.5*p,g=.2+d*1.8*p;r.sprite.scale.set(u,u,1),r.core.scale.set(g,g,1),o.opacity=d*.95,l.opacity=d,o.color.copy(r.color).multiplyScalar(.7+d*.3),r.sprite.position.copy(r.startPos),r.core.position.copy(r.startPos)}else if(r.age<=a){const h=(r.age-r.gestationFrames)/r.flightFrames,d=h<.5?2*h*h:1-Math.pow(-2*h+2,2)/2,p=r.startPos,u=r.lastTargetPos,g=u.x-p.x,_=u.y-p.y,m=u.z-p.z,f=Math.sqrt(g*g+_*_+m*m),T=(p.x+u.x)*.5,E=(p.y+u.y)*.5+30+f*.15,y=(p.z+u.z)*.5,D=1-d,w=D*D,C=2*D*d,I=d*d,S=w*p.x+C*T+I*u.x,M=w*p.y+C*E+I*u.y,A=w*p.z+C*y+I*u.z;r.sprite.position.set(S,M,A),r.core.position.set(S,M,A);const W=1-d*.35;r.sprite.scale.setScalar(5*W),r.core.scale.setScalar(2*W),o.opacity=.95,l.opacity=1,o.color.copy(r.color)}else if(r.arriveFired){const h=r.age-a,d=Math.max(0,1-h/8);o.opacity=.95*d,l.opacity=1*d,r.sprite.scale.setScalar(5*(1+(1-d)*2)),d<=0&&(this.scene.remove(r.sprite),this.scene.remove(r.core),o.dispose(),l.dispose(),this.birthOrbs.splice(s,1))}else{r.arriveFired=!0;try{r.onArrive()}catch(h){console.warn("[birth-orb] onArrive threw",h)}}}}dispose(){for(const t of this.spawnBursts)this.scene.remove(t.particles),t.particles.geometry.dispose(),t.particles.material.dispose();for(const t of this.rainbowBursts)this.scene.remove(t.particles),t.particles.geometry.dispose(),t.particles.material.dispose();for(const t of this.implosions)this.scene.remove(t.particles),t.particles.geometry.dispose(),t.particles.material.dispose(),t.flash&&(this.scene.remove(t.flash),t.flash.geometry.dispose(),t.flash.material.dispose());for(const t of this.shockwaves)this.scene.remove(t.mesh),t.mesh.geometry.dispose(),t.mesh.material.dispose();for(const t of this.connectionFlashes)this.scene.remove(t.line),t.line.geometry.dispose(),t.line.material.dispose();for(const t of this.birthOrbs)this.scene.remove(t.sprite),this.scene.remove(t.core),t.sprite.material.dispose(),t.core.material.dispose();this.pulseEffects=[],this.spawnBursts=[],this.rainbowBursts=[],this.rippleWaves=[],this.implosions=[],this.shockwaves=[],this.connectionFlashes=[],this.birthOrbs=[]}}const wn={bloomStrength:.8,rotateSpeed:.3,fogColor:328976,fogDensity:.008,nebulaIntensity:0,chromaticIntensity:.002,vignetteRadius:.9,breatheAmplitude:1},zn={bloomStrength:1.8,rotateSpeed:.08,fogColor:656672,fogDensity:.006,nebulaIntensity:1,chromaticIntensity:.005,vignetteRadius:.7,breatheAmplitude:2};class Xg{constructor(){kt(this,"active",!1);kt(this,"transition",0);kt(this,"transitionSpeed",.008);kt(this,"current");kt(this,"auroraHue",0);this.current={...wn}}setActive(t){this.active=t}update(t,e,n,s,r){const a=this.active?1:0;this.transition+=(a-this.transition)*this.transitionSpeed*60*(1/60),this.transition=Math.max(0,Math.min(1,this.transition));const o=this.transition;this.current.bloomStrength=this.lerp(wn.bloomStrength,zn.bloomStrength,o),this.current.rotateSpeed=this.lerp(wn.rotateSpeed,zn.rotateSpeed,o),this.current.fogDensity=this.lerp(wn.fogDensity,zn.fogDensity,o),this.current.nebulaIntensity=this.lerp(wn.nebulaIntensity,zn.nebulaIntensity,o),this.current.chromaticIntensity=this.lerp(wn.chromaticIntensity,zn.chromaticIntensity,o),this.current.vignetteRadius=this.lerp(wn.vignetteRadius,zn.vignetteRadius,o),this.current.breatheAmplitude=this.lerp(wn.breatheAmplitude,zn.breatheAmplitude,o),e.strength=this.current.bloomStrength,n.autoRotateSpeed=this.current.rotateSpeed;const l=new ot(wn.fogColor),c=new ot(zn.fogColor),h=l.clone().lerp(c,o);if(t.fog=new Dr(h,this.current.fogDensity),o>.01){this.auroraHue=r*.1%1;const d=new ot().setHSL(.75+this.auroraHue*.15,.8,.5),p=new ot().setHSL(.55+this.auroraHue*.2,.7,.4);s.point1.color.lerp(d,o*.3),s.point2.color.lerp(p,o*.3)}else s.point1.color.set(6514417),s.point2.color.set(11032055)}lerp(t,e,n){return t+(e-t)*n}}const Yg=50,xs=[];function qg(i,t,e){const n=i.tags??[],s=i.type??"";let r=null,a=0;for(const o of t){let l=0;o.type===s&&(l+=2);for(const c of o.tags)n.includes(c)&&(l+=1);l>a&&(a=l,r=o.id)}if(r&&a>0){const o=e.get(r);if(o)return new P(o.x+(Math.random()-.5)*10,o.y+(Math.random()-.5)*10,o.z+(Math.random()-.5)*10)}return new P((Math.random()-.5)*40,(Math.random()-.5)*40,(Math.random()-.5)*40)}function jg(i,t){if(xs.length<=Yg)return;const e=xs.shift();i.edgeManager.removeEdgesForNode(e),i.nodeManager.removeNode(e),i.forceSim.removeNode(e),i.onMutation({type:"edgesRemoved",nodeId:e}),i.onMutation({type:"nodeRemoved",nodeId:e});const n=t.findIndex(s=>s.id===e);n!==-1&&t.splice(n,1)}function Zg(i,t,e){var d,p;const{effects:n,nodeManager:s,edgeManager:r,forceSim:a,camera:o,onMutation:l}=t,c=s.positions,h=s.meshMap;switch(i.type){case"MemoryCreated":{const u=i.data;if(!u.id)break;const g={id:u.id,label:(u.content??"").slice(0,60),type:u.node_type??"fact",retention:Math.max(0,Math.min(1,u.retention??.9)),tags:u.tags??[],createdAt:new Date().toISOString(),updatedAt:new Date().toISOString(),isCenter:!1},_=qg(g,e,c),m=s.addNode(g,_,{isBirthRitual:!0});a.addNode(u.id,m),xs.push(u.id),jg(t,e);const f=new ot(Sa[g.type]||"#00ffd1"),T=f.clone();T.offsetHSL(.15,0,0),n.createBirthOrb(o,f,()=>s.positions.get(g.id),()=>{s.igniteNode(g.id);const E=s.positions.get(g.id)??_,y=s.meshMap.get(g.id);y&&y.scale.multiplyScalar(1.8),n.createRainbowBurst(E,f),n.createShockwave(E,f,o),n.createShockwave(E,T,o),n.createRippleWave(E)}),l({type:"nodeAdded",node:g});break}case"ConnectionDiscovered":{const u=i.data;if(!u.source_id||!u.target_id)break;const g=c.get(u.source_id),_=c.get(u.target_id),m={source:u.source_id,target:u.target_id,weight:u.weight??.5,type:u.connection_type??"semantic"};r.addEdge(m,c),g&&_&&n.createConnectionFlash(g,_,new ot(54527)),u.source_id&&h.has(u.source_id)&&n.addPulse(u.source_id,1,new ot(54527),.02),u.target_id&&h.has(u.target_id)&&n.addPulse(u.target_id,1,new ot(54527),.02),l({type:"edgeAdded",edge:m});break}case"MemoryDeleted":{const u=i.data;if(!u.id)break;const g=c.get(u.id);if(g){const m=new ot(16729943);n.createImplosion(g,m)}r.removeEdgesForNode(u.id),s.removeNode(u.id),a.removeNode(u.id);const _=xs.indexOf(u.id);_!==-1&&xs.splice(_,1),l({type:"edgesRemoved",nodeId:u.id}),l({type:"nodeRemoved",nodeId:u.id});break}case"MemoryPromoted":{const u=i.data,g=u==null?void 0:u.id;if(!g)break;const _=u.new_retention??.95;if(h.has(g)){s.growNode(g,_),n.addPulse(g,1.2,new ot(65416),.01);const m=c.get(g);m&&(n.createShockwave(m,new ot(65416),o),n.createSpawnBurst(m,new ot(65416))),l({type:"nodeUpdated",nodeId:g,retention:_})}break}case"MemoryDemoted":{const u=i.data,g=u==null?void 0:u.id;if(!g)break;const _=u.new_retention??.3;h.has(g)&&(s.growNode(g,_),n.addPulse(g,.8,new ot(16729943),.03),l({type:"nodeUpdated",nodeId:g,retention:_}));break}case"MemoryUpdated":{const u=i.data,g=u==null?void 0:u.id;if(!g||!h.has(g))break;n.addPulse(g,.6,new ot(8490232),.02),u.retention!==void 0&&(s.growNode(g,u.retention),l({type:"nodeUpdated",nodeId:g,retention:u.retention}));break}case"SearchPerformed":{h.forEach((u,g)=>{n.addPulse(g,.6+Math.random()*.4,new ot(8490232),.02)});break}case"DreamStarted":{h.forEach((u,g)=>{n.addPulse(g,1,new ot(11032055),.005)});break}case"DreamProgress":{const u=(d=i.data)==null?void 0:d.memory_id;u&&h.has(u)&&n.addPulse(u,1.5,new ot(12616956),.01);break}case"DreamCompleted":{n.createSpawnBurst(new P(0,0,0),new ot(11032055)),n.createShockwave(new P(0,0,0),new ot(11032055),o);break}case"RetentionDecayed":{const u=(p=i.data)==null?void 0:p.id;u&&h.has(u)&&n.addPulse(u,.8,new ot(16729943),.03);break}case"ConsolidationCompleted":{h.forEach((u,g)=>{n.addPulse(g,.4+Math.random()*.3,new ot(16758784),.015)});break}case"ActivationSpread":{const u=i.data;if(u.source_id&&u.target_ids){const g=c.get(u.source_id);if(g)for(const _ of u.target_ids){const m=c.get(_);m&&n.createConnectionFlash(g,m,new ot(1370310))}}break}case"MemorySuppressed":{const u=i.data;if(!u.id)break;const g=c.get(u.id);if(g){n.createImplosion(g,new ot(11032055));const _=Math.max(1,u.suppression_count??1),m=Math.min(.4+_*.15,1);n.addPulse(u.id,m,new ot(11032055),.04)}break}case"MemoryUnsuppressed":{const u=i.data;if(!u.id)break;const g=c.get(u.id);g&&h.has(u.id)&&(n.createRainbowBurst(g,new ot(65416)),n.addPulse(u.id,1,new ot(65416),.02));break}case"Rac1CascadeSwept":{const g=i.data.neighbors_affected??0;if(g===0)break;const _=Array.from(h.keys()),m=Math.min(g,_.length,12);for(let f=0;f<m;f++){const T=Math.floor(Math.random()*_.length),E=_[T];n.addPulse(E,.5,new ot(11032055),.035)}break}case"Connected":{const u=h.keys().next().value;if(!u)break;const g=c.get(u);g&&n.createRippleWave(g);break}case"ConsolidationStarted":{const u=Array.from(h.keys()),g=Math.min(u.length,20);for(let _=0;_<g;_++){const m=Math.floor(Math.random()*u.length),f=u[m];n.addPulse(f,.45,new ot(16758784),.025)}break}case"ImportanceScored":{const u=i.data;if(!u.id)break;if(c.get(u.id)&&h.has(u.id)){const m=.3+Math.max(0,Math.min(1,u.composite_score??.5))*.7;n.addPulse(u.id,m,new ot(16727212),.03)}break}}}const Kg=`
varying vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = vec4(position, 1.0);
}
`,$g=`
precision highp float;

uniform float uTime;
uniform vec2 uResolution;
uniform float uDreamIntensity;

varying vec2 vUv;

// Simplex-style hash
vec3 hash33(vec3 p3) {
	p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
	p3 += dot(p3, p3.yxz + 33.33);
	return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// 3D value noise
float noise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);

	float n = i.x + i.y * 157.0 + 113.0 * i.z;

	vec4 v1 = fract(sin(vec4(n + 0.0, n + 1.0, n + 157.0, n + 158.0)) * 43758.5453);
	vec4 v2 = fract(sin(vec4(n + 113.0, n + 114.0, n + 270.0, n + 271.0)) * 43758.5453);

	vec4 a = mix(v1, v2, f.z);
	vec2 b = mix(a.xy, a.zw, f.y);
	return mix(b.x, b.y, f.x);
}

// FBM with 5 octaves
float fbm(vec3 p) {
	float value = 0.0;
	float amplitude = 0.5;
	float frequency = 1.0;
	for (int i = 0; i < 5; i++) {
		value += amplitude * noise(p * frequency);
		frequency *= 2.0;
		amplitude *= 0.5;
	}
	return value;
}

// IQ cosine palette
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
	return a + b * cos(6.28318 * (c * t + d));
}

void main() {
	vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / min(uResolution.x, uResolution.y);
	float t = uTime * 0.05;

	// Domain warping: fbm(p + fbm(p + fbm(p)))
	vec3 p = vec3(uv * 2.0, t);

	float warp1 = fbm(p);
	float warp2 = fbm(p + warp1 * 3.0 + vec3(1.7, 9.2, t * 0.3));
	float warp3 = fbm(p + warp2 * 2.5 + vec3(8.3, 2.8, t * 0.2));

	// Final noise value
	float f = fbm(p + warp3 * 2.0);

	// Color: cosmic palette that shifts during dream mode
	vec3 normalA = vec3(0.02, 0.01, 0.05);
	vec3 normalB = vec3(0.03, 0.02, 0.08);
	vec3 normalC = vec3(1.0, 1.0, 1.0);
	vec3 normalD = vec3(0.70, 0.55, 0.80);

	vec3 dreamA = vec3(0.05, 0.01, 0.08);
	vec3 dreamB = vec3(0.06, 0.03, 0.12);
	vec3 dreamC = vec3(1.0, 0.8, 1.0);
	vec3 dreamD = vec3(0.80, 0.40, 0.90);

	vec3 a = mix(normalA, dreamA, uDreamIntensity);
	vec3 b = mix(normalB, dreamB, uDreamIntensity);
	vec3 c = mix(normalC, dreamC, uDreamIntensity);
	vec3 d = mix(normalD, dreamD, uDreamIntensity);

	vec3 color = palette(f + warp2 * 0.5, a, b, c, d);

	// Add subtle star-like highlights
	float stars = smoothstep(0.97, 1.0, noise(vec3(uv * 50.0, t * 0.1)));
	color += stars * 0.15;

	// Intensity modulation
	float intensity = 0.15 + 0.1 * uDreamIntensity;
	color *= intensity;

	// Vignette
	float dist = length(uv);
	color *= smoothstep(1.5, 0.3, dist);

	gl_FragColor = vec4(color, 1.0);
}
`;function Jg(i){const t=new ws(2,2),e=new He({vertexShader:Kg,fragmentShader:$g,uniforms:{uTime:{value:0},uResolution:{value:new St(window.innerWidth,window.innerHeight)},uDreamIntensity:{value:0}},depthWrite:!1,depthTest:!1,transparent:!1}),n=new be(t,e);return n.frustumCulled=!1,n.renderOrder=-1e3,i.add(n),{mesh:n,material:e}}function Qg(i,t,e,n,s){i.uniforms.uTime.value=t,i.uniforms.uDreamIntensity.value=e,i.uniforms.uResolution.value.set(n,s)}const t_={uniforms:{tDiffuse:{value:null},uIntensity:{value:.002}},vertexShader:`
		varying vec2 vUv;
		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
		}
	`,fragmentShader:`
		uniform sampler2D tDiffuse;
		uniform float uIntensity;
		varying vec2 vUv;

		void main() {
			vec2 center = vec2(0.5);
			vec2 dir = vUv - center;
			float dist = length(dir);

			float rOffset = uIntensity * dist;
			float gOffset = 0.0;
			float bOffset = -uIntensity * dist;

			vec2 rUv = vUv + dir * rOffset;
			vec2 gUv = vUv + dir * gOffset;
			vec2 bUv = vUv + dir * bOffset;

			float r = texture2D(tDiffuse, rUv).r;
			float g = texture2D(tDiffuse, gUv).g;
			float b = texture2D(tDiffuse, bUv).b;

			gl_FragColor = vec4(r, g, b, 1.0);
		}
	`},e_={uniforms:{tDiffuse:{value:null},uTime:{value:0},uIntensity:{value:.04}},vertexShader:`
		varying vec2 vUv;
		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
		}
	`,fragmentShader:`
		uniform sampler2D tDiffuse;
		uniform float uTime;
		uniform float uIntensity;
		varying vec2 vUv;

		float rand(vec2 co) {
			return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
		}

		void main() {
			vec4 color = texture2D(tDiffuse, vUv);
			float grain = rand(vUv + vec2(uTime)) * 2.0 - 1.0;
			color.rgb += grain * uIntensity;
			gl_FragColor = color;
		}
	`},n_={uniforms:{tDiffuse:{value:null},uRadius:{value:.9},uSoftness:{value:.5}},vertexShader:`
		varying vec2 vUv;
		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
		}
	`,fragmentShader:`
		uniform sampler2D tDiffuse;
		uniform float uRadius;
		uniform float uSoftness;
		varying vec2 vUv;

		void main() {
			vec4 color = texture2D(tDiffuse, vUv);
			vec2 center = vec2(0.5);
			float dist = distance(vUv, center) * 1.414;
			float vignette = smoothstep(uRadius, uRadius - uSoftness, dist);
			color.rgb *= vignette;
			gl_FragColor = color;
		}
	`};function i_(i){const t=new vr(t_),e=new vr(e_),n=new vr(n_);return i.addPass(t),i.addPass(e),i.addPass(n),{chromatic:t,grain:e,vignette:n}}function s_(i,t,e){i.chromatic.uniforms.uIntensity.value=.002+(.005-.002)*e,i.grain.uniforms.uTime.value=t,i.grain.uniforms.uIntensity.value=.04+e*.02;const r=.9,a=.7;i.vignette.uniforms.uRadius.value=r+(a-r)*e}const Fc="vestige:graph:brightness",Mr=1,go=.5,_o=2.5;function r_(){if(typeof localStorage>"u")return Mr;const i=localStorage.getItem(Fc);if(i===null)return Mr;const t=Number(i);return Number.isFinite(t)?Math.min(_o,Math.max(go,t)):Mr}const ni=a_();function a_(){let i=me(!1),t=me(va(new Date)),e=me(!1),n=me(1),s=me(!1),r=me(va(r_()));return{get temporalEnabled(){return B(i)},set temporalEnabled(a){zt(i,a,!0)},get temporalDate(){return B(t)},set temporalDate(a){zt(t,a,!0)},get temporalPlaying(){return B(e)},set temporalPlaying(a){zt(e,a,!0)},get temporalSpeed(){return B(n)},set temporalSpeed(a){zt(n,a,!0)},get dreamMode(){return B(s)},set dreamMode(a){zt(s,a,!0)},get brightness(){return B(r)},set brightness(a){const o=Math.min(_o,Math.max(go,a));if(zt(r,o,!0),typeof localStorage<"u")try{localStorage.setItem(Fc,String(o))}catch{}},brightnessMin:go,brightnessMax:_o,brightnessDefault:Mr}}var o_=Se('<div class="w-full h-full"></div>');function l_(i,t){ys(t,!0);let e=vs(t,"events",19,()=>[]),n=vs(t,"isDreaming",3,!1),s=vs(t,"colorMode",3,"type");kc(()=>{l==null||l.setColorMode(s())});let r,a,o,l,c,h,d,p,u,g,_,m=null,f=[];jl(()=>{a=Dg(r),g=Jg(a.scene).material,_=i_(a.composer),h=new Gg(a.scene),l=new kg,l.colorMode=s(),c=new Vg,d=new Wg(a.scene),u=new Xg;const M=l.createNodes(t.nodes);c.createEdges(t.edges,M),p=new Ig(M),f=[...t.nodes],a.scene.add(c.group),a.scene.add(l.group),E(),window.addEventListener("resize",D),r.addEventListener("pointermove",w),r.addEventListener("click",C)}),Zl(()=>{cancelAnimationFrame(o),window.removeEventListener("resize",D),r==null||r.removeEventListener("pointermove",w),r==null||r.removeEventListener("click",C),d==null||d.dispose(),h==null||h.dispose(),l==null||l.dispose(),c==null||c.dispose(),a&&Ug(a)});let T=0;function E(){o=requestAnimationFrame(E);const S=performance.now();T===0&&(T=S);const M=S-T;if(M<16)return;T=S-M%16;const A=S*.001;p.tick(t.edges),l.updatePositions(),c.updatePositions(l.positions),c.animateEdges(l.positions),h.animate(A),l.animate(A,f,a.camera,ni.brightness),u.setActive(n()),u.update(a.scene,a.bloomPass,a.controls,a.lights,A),Qg(g,A,u.current.nebulaIntensity,r.clientWidth,r.clientHeight),s_(_,A,u.current.nebulaIntensity),y(),d.update(l.meshMap,a.camera,l.positions),a.controls.update(),a.composer.render()}function y(){if(!e()||e().length===0)return;const S=[];for(const A of e()){if(A===m)break;S.push(A)}if(S.length===0)return;if(S.length===e().length&&e().length>=200){console.warn("[vestige] Event horizon overflow: dropping visuals for",S.length,"events"),m=e()[0];return}m=e()[0];const M={effects:d,nodeManager:l,edgeManager:c,forceSim:p,camera:a.camera,onMutation:A=>{var W;A.type==="nodeAdded"?f=[...f,A.node]:A.type==="nodeRemoved"&&(f=f.filter(k=>k.id!==A.nodeId)),(W=t.onGraphMutation)==null||W.call(t,A)}};for(let A=S.length-1;A>=0;A--)Zg(S[A],M,f)}function D(){!r||!a||Lg(a,r)}function w(S){const M=r.getBoundingClientRect();a.mouse.x=(S.clientX-M.left)/M.width*2-1,a.mouse.y=-((S.clientY-M.top)/M.height)*2+1,a.raycaster.setFromCamera(a.mouse,a.camera);const A=a.raycaster.intersectObjects(l.getMeshes());A.length>0?(l.hoveredNode=A[0].object.userData.nodeId,r.style.cursor="pointer"):(l.hoveredNode=null,r.style.cursor="grab")}function C(){var S;if(l.hoveredNode){l.selectedNode=l.hoveredNode,(S=t.onSelect)==null||S.call(t,l.hoveredNode);const M=l.positions.get(l.hoveredNode);M&&a.controls.target.lerp(M.clone(),.5)}}var I=o_();Yc(I,S=>r=S,()=>r),_e(i,I),Es()}var c_=Se('<div class="flex items-center gap-1 text-[10px]"><span class="text-muted"> </span> <span> </span></div>'),h_=Se('<div class="space-y-2"><svg class="w-full"><line x1="4" stroke="#2a2a5e" stroke-width="0.5" stroke-dasharray="2,4"></line><line x1="4" stroke="#ef444430" stroke-width="0.5" stroke-dasharray="2,4"></line><path fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round"></path><path fill="url(#curveGrad)" opacity="0.15"></path><circle cx="4" r="3"></circle><defs><linearGradient id="curveGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#6366f1"></stop><stop offset="100%" stop-color="#6366f100"></stop></linearGradient></defs></svg> <div class="flex gap-2 flex-wrap"></div></div>');function u_(i,t){ys(t,!0);let e=vs(t,"width",3,240),n=vs(t,"height",3,80);function s(m){return t.stability<=0?0:Math.exp(-m/t.stability)}let r=Gn(()=>{const m=[],f=Math.max(t.stability*3,30),T=4,E=e()-T*2,y=n()-T*2;for(let D=0;D<=50;D++){const w=D/50*f,C=s(w),I=T+D/50*E,S=T+(1-C)*y;m.push(`${D===0?"M":"L"}${I.toFixed(1)},${S.toFixed(1)}`)}return m.join(" ")}),a=Gn(()=>[{label:"Now",days:0,value:t.retention},{label:"1d",days:1,value:s(1)},{label:"7d",days:7,value:s(7)},{label:"30d",days:30,value:s(30)}]);function o(m){return m>.7?"#10b981":m>.4?"#f59e0b":"#ef4444"}var l=h_(),c=yt(l),h=yt(c),d=bt(h),p=bt(d),u=bt(p),g=bt(u);Hc(),xt(c);var _=bt(c,2);_s(_,21,()=>B(a),hr,(m,f)=>{var T=c_(),E=yt(T),y=yt(E);xt(E);var D=bt(E,2),w=yt(D);xt(D),xt(T),Ke((C,I)=>{fe(y,`${B(f).label??""}:`),Sr(D,`color: ${C??""}`),fe(w,`${I??""}%`)},[()=>o(B(f).value),()=>(B(f).value*100).toFixed(0)]),_e(m,T)}),xt(_),xt(l),Ke(m=>{ve(c,"width",e()),ve(c,"height",n()),ve(c,"viewBox",`0 0 ${e()??""} ${n()??""}`),ve(h,"y1",4+(n()-8)*.5),ve(h,"x2",e()-4),ve(h,"y2",4+(n()-8)*.5),ve(d,"y1",4+(n()-8)*.8),ve(d,"x2",e()-4),ve(d,"y2",4+(n()-8)*.8),ve(p,"d",B(r)),ve(u,"d",`${B(r)??""} L${e()-4},${n()-4} L4,${n()-4} Z`),ve(g,"cy",4+(1-t.retention)*(n()-8)),ve(g,"fill",m)},[()=>o(t.retention)]),_e(i,l),Es()}function ql(i,t,e){const n=e.getTime(),s=new Set,r=new Map,a=i.filter(l=>{const c=new Date(l.createdAt).getTime();if(c<=n){s.add(l.id);const h=n-c,d=1440*60*1e3,p=h<d?.3+.7*(h/d):1;return r.set(l.id,p),!0}return!1}),o=t.filter(l=>s.has(l.source)&&s.has(l.target));return{visibleNodes:a,visibleEdges:o,nodeOpacities:r}}function d_(i){if(i.length===0){const n=new Date;return{oldest:n,newest:n}}let t=1/0,e=-1/0;for(const n of i){const s=new Date(n.createdAt).getTime();s<t&&(t=s),s>e&&(e=s)}return{oldest:new Date(t),newest:new Date(e)}}var f_=Se(`<div class="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 w-[90%] max-w-xl"><div class="glass-panel rounded-xl p-3 space-y-2"><div class="flex items-center justify-between"><div class="flex items-center gap-2"><button class="w-7 h-7 rounded-lg bg-synapse/20 border border-synapse/30 text-synapse-glow text-xs flex items-center justify-center hover:bg-synapse/30 transition"> </button> <select class="px-2 py-1 bg-white/[0.03] border border-synapse/10 rounded-lg text-[10px] text-dim focus:outline-none"><option>1x</option><option>7x</option><option>30x</option></select></div> <span class="text-xs text-bright font-medium"> </span> <button class="text-[10px] text-muted hover:text-text transition">Close</button></div> <input type="range" min="0" max="100" step="0.1" class="w-full h-1.5 appearance-none bg-white/[0.06] rounded-full cursor-pointer
					[&amp;::-webkit-slider-thumb]:appearance-none [&amp;::-webkit-slider-thumb]:w-3 [&amp;::-webkit-slider-thumb]:h-3
					[&amp;::-webkit-slider-thumb]:rounded-full [&amp;::-webkit-slider-thumb]:bg-synapse-glow
					[&amp;::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(129,140,248,0.4)]"/> <div class="flex justify-between text-[9px] text-muted"><span> </span> <span> </span></div></div></div>`),p_=Se('<button class="absolute bottom-4 right-4 z-10 px-3 py-2 glass rounded-xl text-dim text-xs hover:text-text transition flex items-center gap-1.5"><span>◷</span> <span>Timeline</span></button>');function m_(i,t){ys(t,!0);let e=me(!1),n=me(!1),s=me(1),r=me(100),a,o=0,l=Gn(()=>d_(t.nodes)),c=Gn(()=>{const E=B(l).oldest.getTime(),D=B(l).newest.getTime()-E||1;return new Date(E+B(r)/100*D)});function h(E){return E.toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}function d(){zt(e,!B(e)),t.onToggle(B(e)),B(e)&&(zt(r,100),t.onDateChange(B(c)))}function p(){zt(n,!B(n)),B(n)?(zt(r,0),o=performance.now(),u()):cancelAnimationFrame(a)}function u(){B(n)&&(a=requestAnimationFrame(E=>{const y=(E-o)/1e3;o=E;const D=B(l).oldest.getTime(),C=(B(l).newest.getTime()-D)/(1440*60*1e3)||1,I=B(s)/C*100;if(zt(r,Math.min(100,B(r)+I*y),!0),t.onDateChange(B(c)),B(r)>=100){zt(n,!1);return}u()}))}function g(){t.onDateChange(B(c))}Zl(()=>{zt(n,!1),cancelAnimationFrame(a)});var _=Gc(),m=Kl(_);{var f=E=>{var y=f_(),D=yt(y),w=yt(D),C=yt(w),I=yt(C),S=yt(I,!0);xt(I);var M=bt(I,2),A=yt(M);A.value=A.__value=1;var W=bt(A);W.value=W.__value=7;var k=bt(W);k.value=k.__value=30,xt(M),xt(C);var q=bt(C,2),Q=yt(q,!0);xt(q);var X=bt(q,2);xt(w);var tt=bt(w,2);xa(tt);var H=bt(tt,2),st=yt(H),gt=yt(st,!0);xt(st);var Et=bt(st,2),Ft=yt(Et,!0);xt(Et),xt(H),xt(D),xt(y),Ke((Xt,Y,nt)=>{fe(S,B(n)?"⏸":"▶"),fe(Q,Xt),fe(gt,Y),fe(Ft,nt)},[()=>h(B(c)),()=>h(B(l).oldest),()=>h(B(l).newest)]),Fe("click",I,p),Jl(M,()=>B(s),Xt=>zt(s,Xt)),Fe("click",X,d),Fe("input",tt,g),Ma(tt,()=>B(r),Xt=>zt(r,Xt)),_e(E,y)},T=E=>{var y=p_();Fe("click",y,d),_e(E,y)};kn(m,E=>{B(e)?E(f):E(T,!1)})}_e(i,_),Es()}$l(["click","input"]);var g_=Se('<div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full flex-shrink-0"></span> <span class="text-text capitalize"> </span> <span class="text-muted text-[10px] ml-auto"> </span></div>'),__=Se('<div class="pointer-events-auto glass-subtle rounded-xl px-3 py-2.5 text-xs space-y-1.5 backdrop-blur-md border border-synapse/10" role="group" aria-label="Memory state colour legend"><div class="text-[10px] uppercase tracking-wider text-muted font-semibold mb-1.5">FSRS accessibility</div> <!></div>');function v_(i,t){ys(t,!1);const e=["active","dormant","silent","unavailable"];qc();var n=__(),s=bt(yt(n),2);_s(s,1,()=>e,r=>r,(r,a)=>{var o=g_(),l=yt(o),c=bt(l,2),h=yt(c,!0);xt(c);var d=bt(c,2),p=yt(d,!0);xt(d),xt(o),Ke(u=>{Sr(l,`background: ${po[B(a)]??""}; box-shadow: 0 0 6px ${po[B(a)]??""}55;`),fe(h,B(a)),fe(p,u)},[()=>{var u;return((u=Fg[B(a)].match(/\(([^)]+)\)/))==null?void 0:u[1])??""}]),_e(r,o)}),xt(n),_e(i,n),Es()}var x_=Se('<div class="h-full flex items-center justify-center"><div class="text-center space-y-4"><div class="w-16 h-16 mx-auto rounded-full border-2 border-synapse/30 border-t-synapse animate-spin"></div> <p class="text-dim text-sm">Loading memory graph...</p></div></div>'),M_=Se(`<div class="h-full flex items-center justify-center"><div class="text-center space-y-5 max-w-lg px-8"><div class="text-5xl opacity-40">⚡</div> <h2 class="text-xl text-bright">MCP Backend Offline</h2> <p class="text-dim text-sm leading-relaxed">The Vestige MCP server isn't reachable on <code class="font-mono text-muted">:3927</code>.
					The dashboard is running but has nothing to query.</p> <div class="glass-subtle rounded-xl p-4 text-left text-xs font-mono text-dim space-y-2"><div class="text-muted text-[10px] uppercase tracking-wider">Start the backend:</div> <code class="block whitespace-pre-wrap break-all text-text">nohup bash -c 'tail -f /dev/null | VESTIGE_DASHBOARD_ENABLED=true ~/.local/bin/vestige-mcp' &gt; /tmp/vestige.log 2&gt;&amp;1 &amp;
disown</code></div> <div class="flex gap-2 justify-center"><button class="px-4 py-2 bg-synapse/20 border border-synapse/40 text-synapse-glow text-xs rounded-xl hover:bg-synapse/30 transition">Retry</button> <a class="px-4 py-2 bg-dream/20 border border-dream/40 text-dream-glow text-xs rounded-xl hover:bg-dream/30 transition">Try demos (no backend needed)</a></div></div></div>`),S_=Se('<div class="h-full flex items-center justify-center"><div class="text-center space-y-4 max-w-md px-8"><div class="text-5xl opacity-30">◎</div> <h2 class="text-xl text-bright">Your Mind Awaits</h2> <p class="text-dim text-sm">No memories yet. Start using Vestige to populate your graph.</p></div></div>'),y_=Se('<div class="h-full flex items-center justify-center"><div class="text-center space-y-4 max-w-md px-8"><div class="text-5xl opacity-30">◎</div> <h2 class="text-xl text-bright">Your Mind Awaits</h2> <p class="text-dim text-sm"> </p></div></div>'),E_=Se('<span> </span> <span class="mx-2 text-subtle">·</span> <span> </span> <span class="mx-2 text-subtle">·</span> <span> </span>',1),b_=Se('<div class="absolute bottom-4 right-4 z-10"><!></div>'),T_=Se('<div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full"></span> <span class="text-dim"> </span></div>'),w_=Se('<div class="absolute bottom-4 right-4 z-10 glass rounded-xl px-4 py-3 text-xs"><div class="text-bright font-semibold mb-2">AhaGraph</div> <div class="space-y-1.5"></div></div>'),A_=Se('<span class="px-2 py-0.5 rounded-lg text-xs bg-white/[0.04] text-dim"> </span>'),R_=Se('<div><div class="flex justify-between text-xs text-dim mb-0.5"><span> </span> <span> </span></div> <div class="h-1.5 bg-white/[0.04] rounded-full overflow-hidden"><div class="h-full rounded-full transition-all duration-500"></div></div></div>'),C_=Se("<div> </div>"),P_=Se(`<div class="absolute right-0 top-0 h-full w-96 glass-panel p-6 overflow-y-auto z-20
			transition-transform duration-300"><div class="flex justify-between items-start mb-4"><h3 class="text-bright text-sm font-semibold">Memory Detail</h3> <button class="text-dim hover:text-text text-lg leading-none">×</button></div> <div class="space-y-4"><div class="flex gap-2 flex-wrap"><span class="px-2 py-0.5 rounded-lg text-xs bg-synapse/20 text-synapse-glow"> </span> <!></div> <div class="text-sm text-text leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto"> </div> <div class="space-y-2"></div> <div><div class="text-xs text-dim mb-1 font-medium">Retention Forecast</div> <!></div> <div class="text-xs text-muted space-y-1"><div> </div> <div> </div> <!> <div> </div></div> <div class="flex gap-2 pt-2"><button class="flex-1 px-3 py-2 rounded-xl bg-recall/20 text-recall text-xs hover:bg-recall/30 transition">↑ Promote</button> <button class="flex-1 px-3 py-2 rounded-xl bg-decay/20 text-decay text-xs hover:bg-decay/30 transition">↓ Demote</button></div> <a class="block text-center px-3 py-2 rounded-xl bg-dream/10 text-dream-glow text-xs hover:bg-dream/20 transition border border-dream/20">◬ Explore Connections</a></div></div>`),D_=Se(`<div class="h-full relative"><!> <div class="absolute top-4 left-4 right-4 z-10 flex items-center gap-3"><div class="flex gap-2 flex-1 max-w-md"><input type="text" placeholder="Center graph on..." class="flex-1 px-3 py-2 glass rounded-xl text-text text-sm
					placeholder:text-muted focus:outline-none focus:!border-synapse/40 transition"/> <button class="px-3 py-2 bg-synapse/20 border border-synapse/40 text-synapse-glow text-sm rounded-xl hover:bg-synapse/30 transition backdrop-blur-sm">Focus</button></div> <div class="flex gap-2 ml-auto"><div class="flex glass rounded-xl p-0.5 text-xs" role="radiogroup" aria-label="Colour mode"><button type="button" role="radio" title="Colour by node type (fact, concept, event, …)">Type</button> <button type="button" role="radio" title="Colour by FSRS memory state (active / dormant / silent / unavailable)">State</button> <button type="button" role="radio" title="Colour by AhaGraph tags (aha / confusion / failure)">AhaGraph</button></div> <select class="px-2 py-2 glass rounded-xl text-dim text-xs"><option>50 nodes</option><option>100 nodes</option><option>150 nodes</option><option>200 nodes</option></select> <label class="flex items-center gap-2 px-3 py-2 glass rounded-xl text-dim text-xs select-none"><span class="text-synapse-glow">☀</span> <input type="range" step="0.1" class="w-20 accent-synapse cursor-pointer" aria-label="Graph brightness"/> <span class="font-mono text-[10px] text-muted w-8 text-right"> </span></label> <button> </button> <button class="px-3 py-2 glass rounded-xl text-dim text-sm hover:text-text transition">↻</button></div></div> <div class="absolute bottom-4 left-4 z-10 text-xs text-dim glass rounded-xl px-3 py-2"><!></div> <!> <!> <!> <!></div>`);function Q_(i,t){ys(t,!0);const e=()=>Xc(jc,"$eventFeed",n),[n,s]=Wc();let r=me(null),a=me(null),o=me(!0),l=me(""),c=me(!1),h=me(""),d=me(150),p=me(!1),u=me(va(new Date)),g=me("type");const _=Object.entries(xr);let m=me(0),f=me(0),T=Gn(()=>B(r)?B(p)?ql(B(r).nodes,B(r).edges,B(u)).visibleNodes:B(r).nodes:[]),E=Gn(()=>B(r)?B(p)?ql(B(r).nodes,B(r).edges,B(u)).visibleEdges:B(r).edges:[]);function y(V){if(B(r))switch(V.type){case"nodeAdded":B(r).nodes=[...B(r).nodes,V.node],B(r).nodeCount=B(r).nodes.length,zt(m,B(r).nodeCount,!0);break;case"nodeRemoved":B(r).nodes=B(r).nodes.filter(at=>at.id!==V.nodeId),B(r).nodeCount=B(r).nodes.length,zt(m,B(r).nodeCount,!0);break;case"edgeAdded":B(r).edges=[...B(r).edges,V.edge],B(r).edgeCount=B(r).edges.length,zt(f,B(r).edgeCount,!0);break;case"edgesRemoved":B(r).edges=B(r).edges.filter(at=>at.source!==V.nodeId&&at.target!==V.nodeId),B(r).edgeCount=B(r).edges.length,zt(f,B(r).edgeCount,!0);break;case"nodeUpdated":{const at=B(r).nodes.find(Z=>Z.id===V.nodeId);at&&(at.retention=V.retention);break}}}jl(()=>{const V=new URLSearchParams(window.location.search).get("colorMode");D(V)&&zt(g,V,!0),w()});function D(V){return V==="type"||V==="state"||V==="ahagraph"}async function w(V,at){var Z;zt(o,!0),zt(l,"");try{const it=!V&&!at;if(zt(r,await gi.graph({max_nodes:B(d),depth:3,query:V||void 0,center_id:at||void 0,sort:it?"recent":void 0}),!0),it&&B(r)&&B(r).nodeCount<=1&&B(r).edgeCount===0){const pt=await gi.graph({max_nodes:B(d),depth:3,sort:"connected"});pt&&pt.nodeCount>B(r).nodeCount&&zt(r,pt,!0)}B(r)&&(zt(m,B(r).nodeCount,!0),zt(f,B(r).edgeCount,!0))}catch(it){const pt=it instanceof Error?it.message:String(it),wt=pt.replace(/\/[\w./-]+\.(sqlite|rs|db|toml|lock)\b/g,"[path]").slice(0,200),ht=it instanceof TypeError||/failed to fetch|NetworkError|load failed/i.test(pt)||/^API 500:?\s*(Internal Server Error)?\s*$/i.test(pt.trim()),Bt=(((Z=B(r))==null?void 0:Z.nodeCount)??0)===0&&/not found|404|empty|no memor/i.test(pt);ht?zt(l,"OFFLINE"):Bt?zt(l,"EMPTY"):zt(l,`Failed to load graph: ${wt}`)}finally{zt(o,!1)}}async function C(){zt(c,!0);try{await gi.dream(),await w()}catch{}finally{zt(c,!1)}}async function I(V){try{zt(a,await gi.memories.get(V),!0)}catch{zt(a,null)}}function S(){B(h).trim()&&w(B(h))}var M=D_(),A=yt(M);{var W=V=>{var at=x_();_e(V,at)},k=V=>{var at=M_(),Z=yt(at),it=bt(yt(Z),8),pt=yt(it),wt=bt(pt,2);xt(it),xt(Z),xt(at),Ke(()=>ve(wt,"href",`${Do??""}/settings`)),Fe("click",pt,()=>w()),_e(V,at)},q=V=>{var at=S_();_e(V,at)},Q=V=>{var at=y_(),Z=yt(at),it=bt(yt(Z),4),pt=yt(it,!0);xt(it),xt(Z),xt(at),Ke(()=>fe(pt,B(l))),_e(V,at)},X=V=>{l_(V,{get nodes(){return B(T)},get edges(){return B(E)},get centerId(){return B(r).center_id},get events(){return e()},get isDreaming(){return B(c)},get colorMode(){return B(g)},onSelect:I,onGraphMutation:y})};kn(A,V=>{B(o)?V(W):B(l)==="OFFLINE"?V(k,1):B(l)==="EMPTY"?V(q,2):B(l)?V(Q,3):B(r)&&V(X,4)})}var tt=bt(A,2),H=yt(tt),st=yt(H);xa(st);var gt=bt(st,2);xt(H);var Et=bt(H,2),Ft=yt(Et),Xt=yt(Ft),Y=bt(Xt,2),nt=bt(Y,2);xt(Ft);var _t=bt(Ft,2),lt=yt(_t);lt.value=lt.__value=50;var Ct=bt(lt);Ct.value=Ct.__value=100;var Lt=bt(Ct);Lt.value=Lt.__value=150;var Vt=bt(Lt);Vt.value=Vt.__value=200,xt(_t);var ie=bt(_t,2),Wt=bt(yt(ie),2);xa(Wt);var ue=bt(Wt,2),R=yt(ue);xt(ue),xt(ie);var ye=bt(ie,2),qt=yt(ye,!0);xt(ye);var jt=bt(ye,2);xt(Et),xt(tt);var At=bt(tt,2),le=yt(At);{var Rt=V=>{var at=E_(),Z=Kl(at),it=yt(Z);xt(Z);var pt=bt(Z,4),wt=yt(pt);xt(pt);var ht=bt(pt,4),Bt=yt(ht);xt(ht),Ke(()=>{fe(it,`${B(m)??""} nodes`),fe(wt,`${B(f)??""} edges`),fe(Bt,`depth ${B(r).depth??""}`)}),_e(V,at)};kn(le,V=>{B(r)&&V(Rt)})}xt(At);var b=bt(At,2);{var v=V=>{var at=b_(),Z=yt(at);v_(Z,{}),xt(at),_e(V,at)};kn(b,V=>{B(g)==="state"&&V(v)})}var F=bt(b,2);{var K=V=>{var at=w_(),Z=bt(yt(at),2);_s(Z,21,()=>_,hr,(it,pt)=>{var wt=Gn(()=>Vc(B(pt),2));let ht=()=>B(wt)[0],Bt=()=>B(wt)[1];var Ut=T_(),Zt=yt(Ut),L=bt(Zt,2),rt=yt(L,!0);xt(L),xt(Ut),Ke(()=>{Sr(Zt,`background: ${Bt()??""}`),fe(rt,Og[ht()])}),_e(it,Ut)}),xt(Z),xt(at),_e(V,at)};kn(F,V=>{B(g)==="ahagraph"&&V(K)})}var J=bt(F,2);{var j=V=>{m_(V,{get nodes(){return B(r).nodes},onDateChange:at=>{zt(u,at,!0)},onToggle:at=>{zt(p,at,!0)}})};kn(J,V=>{B(r)&&V(j)})}var Tt=bt(J,2);{var dt=V=>{var at=P_(),Z=yt(at),it=bt(yt(Z),2);xt(Z);var pt=bt(Z,2),wt=yt(pt),ht=yt(wt),Bt=yt(ht,!0);xt(ht);var Ut=bt(ht,2);_s(Ut,17,()=>B(a).tags,hr,(Ve,Re)=>{var tn=A_(),en=yt(tn,!0);xt(tn),Ke(()=>fe(en,B(Re))),_e(Ve,tn)}),xt(wt);var Zt=bt(wt,2),L=yt(Zt,!0);xt(Zt);var rt=bt(Zt,2);_s(rt,21,()=>[{label:"Retention",value:B(a).retentionStrength},{label:"Storage",value:B(a).storageStrength},{label:"Retrieval",value:B(a).retrievalStrength}],hr,(Ve,Re)=>{var tn=R_(),en=yt(tn),pi=yt(en),Ps=yt(pi,!0);xt(pi);var Ds=bt(pi,2),Ir=yt(Ds);xt(Ds),xt(en);var Ls=bt(en,2),Nr=yt(Ls);xt(Ls),xt(tn),Ke(Fr=>{fe(Ps,B(Re).label),fe(Ir,`${Fr??""}%`),Sr(Nr,`width: ${B(Re).value*100}%; background: ${B(Re).value>.7?"#10b981":B(Re).value>.4?"#f59e0b":"#ef4444"}`)},[()=>(B(Re).value*100).toFixed(1)]),_e(Ve,tn)}),xt(rt);var G=bt(rt,2),$=bt(yt(G),2);{let Ve=Gn(()=>B(a).storageStrength*30);u_($,{get retention(){return B(a).retentionStrength},get stability(){return B(Ve)}})}xt(G);var ft=bt(G,2),ut=yt(ft),Ot=yt(ut);xt(ut);var ce=bt(ut,2),Ae=yt(ce);xt(ce);var $t=bt(ce,2);{var Ye=Ve=>{var Re=C_(),tn=yt(Re);xt(Re),Ke(en=>fe(tn,`Accessed: ${en??""}`),[()=>new Date(B(a).lastAccessedAt).toLocaleString()]),_e(Ve,Re)};kn($t,Ve=>{B(a).lastAccessedAt&&Ve(Ye)})}var Qe=bt($t,2),Rs=yt(Qe);xt(Qe),xt(ft);var fi=bt(ft,2),pn=yt(fi),ns=bt(pn,2);xt(fi);var Cs=bt(fi,2);xt(pt),xt(at),Ke((Ve,Re)=>{fe(Bt,B(a).nodeType),fe(L,B(a).content),fe(Ot,`Created: ${Ve??""}`),fe(Ae,`Updated: ${Re??""}`),fe(Rs,`Reviews: ${B(a).reviewCount??0??""}`),ve(Cs,"href",`${Do??""}/explore`)},[()=>new Date(B(a).createdAt).toLocaleString(),()=>new Date(B(a).updatedAt).toLocaleString()]),Fe("click",it,()=>zt(a,null)),Fe("click",pn,()=>{B(a)&&gi.memories.promote(B(a).id)}),Fe("click",ns,()=>{B(a)&&gi.memories.demote(B(a).id)}),_e(V,at)};kn(Tt,V=>{B(a)&&V(dt)})}xt(M),Ke((V,at)=>{ve(Xt,"aria-checked",B(g)==="type"),Us(Xt,1,`px-3 py-1.5 rounded-lg transition ${B(g)==="type"?"bg-synapse/25 text-synapse-glow":"text-dim hover:text-text"}`),ve(Y,"aria-checked",B(g)==="state"),Us(Y,1,`px-3 py-1.5 rounded-lg transition ${B(g)==="state"?"bg-synapse/25 text-synapse-glow":"text-dim hover:text-text"}`),ve(nt,"aria-checked",B(g)==="ahagraph"),Us(nt,1,`px-3 py-1.5 rounded-lg transition ${B(g)==="ahagraph"?"bg-synapse/25 text-synapse-glow":"text-dim hover:text-text"}`),ve(ie,"title",`Adjust graph brightness (${V??""}x). Combines with auto distance compensation.`),ve(Wt,"min",ni.brightnessMin),ve(Wt,"max",ni.brightnessMax),fe(R,`${at??""}x`),ye.disabled=B(c),Us(ye,1,`px-4 py-2 rounded-xl bg-dream/20 border border-dream/40 text-dream-glow text-sm
					hover:bg-dream/30 transition-all backdrop-blur-sm disabled:opacity-50
					${B(c)?"glow-dream animate-pulse-glow":""}`),fe(qt,B(c)?"◈ Dreaming...":"◈ Dream")},[()=>ni.brightness.toFixed(1),()=>ni.brightness.toFixed(1)]),Fe("keydown",st,V=>V.key==="Enter"&&S()),Ma(st,()=>B(h),V=>zt(h,V)),Fe("click",gt,S),Fe("click",Xt,()=>zt(g,"type")),Fe("click",Y,()=>zt(g,"state")),Fe("click",nt,()=>zt(g,"ahagraph")),Fe("change",_t,()=>w()),Jl(_t,()=>B(d),V=>zt(d,V)),Ma(Wt,()=>ni.brightness,V=>ni.brightness=V),Fe("click",ye,C),Fe("click",jt,()=>w()),_e(i,M),Es(),s()}$l(["click","keydown","change"]);export{Q_ as component};
