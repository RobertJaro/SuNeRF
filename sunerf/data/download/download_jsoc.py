import os


def donwload_ds(ds, dir, client, process=None):
    os.makedirs(dir, exist_ok=True)
    r = client.export(ds, protocol='fits', process=process)
    r.wait()
    download_result = r.download(dir)
    return download_result
