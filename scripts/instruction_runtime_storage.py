"""Native read-only pagefile telemetry plus the unchanged disk reserve guard."""
import ctypes
from ctypes import wintypes
import mmap
from instruction_disk_guard import snapshot, require_safe

class PagefileInfo(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('reserved',wintypes.DWORD),
              ('total_pages',ctypes.c_size_t),('used_pages',ctypes.c_size_t),('peak_pages',ctypes.c_size_t)]

def runtime_snapshot():
    record=snapshot()
    rows=[]
    callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,ctypes.c_void_p,ctypes.POINTER(PagefileInfo),wintypes.LPCWSTR)
    def receive(context,info,name):
        x=info.contents
        rows.append(dict(path=name,allocated_bytes=x.total_pages*mmap.PAGESIZE,
                         used_bytes=x.used_pages*mmap.PAGESIZE,peak_used_bytes=x.peak_pages*mmap.PAGESIZE))
        return True
    callback=callback_type(receive)
    api=ctypes.WinDLL('psapi',use_last_error=True).EnumPageFilesW
    api.argtypes=[callback_type,ctypes.c_void_p];api.restype=wintypes.BOOL
    if not api(callback,None):raise ctypes.WinError(ctypes.get_last_error())
    record['pagefiles']=rows
    record['pagefile_api']='EnumPageFilesW; page counts multiplied by system page size'
    return record
