"""
Oyun penceresini (pencere ustte olmasa bile) X11 uzerinden yakalar.

GNOME/Mutter gibi bilesimci (compositor) altinda XGetImage, ortulu pencerenin
icerigini de dondurur. Wine'daki Unreal penceresini adiyla bulur.

    cap = WindowCapture()          # "DronesOfWar" penceresini bulur
    frame = cap.grab()             # BGR uint8 (H, W, 3), siyah bantlar kirpilmis
"""
import ctypes
import ctypes.util
import os

import numpy as np

_xlib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("X11"))


class _XImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int),
        ("format", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_char)),
        ("byte_order", ctypes.c_int), ("bitmap_unit", ctypes.c_int),
        ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int),
        ("depth", ctypes.c_int), ("bytes_per_line", ctypes.c_int),
        ("bits_per_pixel", ctypes.c_int),
    ]


_xlib.XOpenDisplay.restype = ctypes.c_void_p
_xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
_xlib.XDefaultRootWindow.restype = ctypes.c_ulong
_xlib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
_xlib.XGetImage.restype = ctypes.POINTER(_XImage)
_xlib.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                            ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
_xlib.XDestroyImage = None  # makro; yerine XFree kullanilir (asagida)
_xlib.XFree.argtypes = [ctypes.c_void_p]
_xlib.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                             ctypes.POINTER(ctypes.c_ulong),
                             ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                             ctypes.POINTER(ctypes.c_uint)]
_xlib.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]


class _XWindowAttributes(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int),
                ("height", ctypes.c_int), ("border_width", ctypes.c_int),
                ("depth", ctypes.c_int), ("visual", ctypes.c_void_p),
                ("root", ctypes.c_ulong), ("_pad", ctypes.c_char * 200)]


_xlib.XGetWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                       ctypes.POINTER(_XWindowAttributes)]

ZPixmap = 2
AllPlanes = 0xFFFFFFFF


class WindowCapture:
    def __init__(self, title_substr="DronesOfWar", display=None, crop_black_bars=True):
        disp = (display or os.environ.get("DISPLAY", ":1")).encode()
        self.dpy = _xlib.XOpenDisplay(disp)
        if not self.dpy:
            raise RuntimeError("X display acilamadi")
        self.win = self._find(_xlib.XDefaultRootWindow(self.dpy), title_substr)
        if not self.win:
            raise RuntimeError(f"'{title_substr}' penceresi bulunamadi")
        attr = _XWindowAttributes()
        _xlib.XGetWindowAttributes(self.dpy, self.win, ctypes.byref(attr))
        self.width, self.height = attr.width, attr.height
        self.crop = None
        if crop_black_bars:
            f = self._grab_raw()
            rows = np.where(f.reshape(f.shape[0], -1).max(axis=1) > 12)[0]
            if len(rows) > 10:
                self.crop = (int(rows[0]), int(rows[-1]) + 1)

    def _find(self, w, sub):
        name = ctypes.c_char_p()
        if _xlib.XFetchName(self.dpy, w, ctypes.byref(name)) and name.value:
            n = name.value.decode(errors="ignore")
            _xlib.XFree(name)
            if n.strip() == sub:
                return w
        root, parent = ctypes.c_ulong(), ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        n = ctypes.c_uint()
        if not _xlib.XQueryTree(self.dpy, w, ctypes.byref(root), ctypes.byref(parent),
                                ctypes.byref(children), ctypes.byref(n)):
            return 0
        kids = [children[i] for i in range(n.value)]
        if children:
            _xlib.XFree(children)
        for k in kids:
            r = self._find(k, sub)
            if r:
                return r
        return 0

    def _grab_raw(self):
        img = _xlib.XGetImage(self.dpy, self.win, 0, 0, self.width, self.height, AllPlanes, ZPixmap)
        if not img:
            raise RuntimeError("XGetImage basarisiz (pencere kapandi mi?)")
        im = img.contents
        buf = ctypes.string_at(im.data, im.bytes_per_line * im.height)
        a = np.frombuffer(buf, np.uint8).reshape(im.height, im.bytes_per_line // 4, 4)
        a = a[:, :im.width, :3].copy()
        # XImage yapisini ve verisini serbest birak (XDestroyImage makrosunun karsiligi)
        _xlib.XFree(im.data)
        _xlib.XFree(img)
        return a

    def grab(self):
        f = self._grab_raw()
        if self.crop:
            f = f[self.crop[0]:self.crop[1]]
        return f
