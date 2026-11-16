"""Fetch the declared author implementations, verifying each source before use."""
from pathlib import Path
import json,hashlib,urllib.request,urllib.parse,base64
root=Path(__file__).resolve().parent
for item in json.loads((root/'third_party_sources.json').read_text()):
    destination=root/'third_party'/item['destination']
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Existing file differs: '+str(destination))
        continue
    raw='https://raw.githubusercontent.com/{repository}/{ref}/{source}'.format(**item)
    try:
        request=urllib.request.Request(raw,headers={'User-Agent':'PIDS-CCF-reproduction'})
        with urllib.request.urlopen(request,timeout=30) as response:content=response.read()
    except Exception:
        api='https://api.github.com/repos/{repository}/contents/{source}?ref={ref}'.format(**item)
        request=urllib.request.Request(api,headers={'User-Agent':'PIDS-CCF-reproduction'})
        with urllib.request.urlopen(request,timeout=30) as response:payload=json.load(response)
        content=base64.b64decode(payload['content'])
    if hashlib.sha256(content).hexdigest()!=item['sha256']:
        raise RuntimeError('Downloaded source does not match recorded implementation: '+item['destination'])
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(content)
    print('Verified',item['destination'])
