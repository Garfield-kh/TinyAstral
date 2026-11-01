"""Audit the exact serialized request text and image digests without altering policy input."""
import copy, hashlib, json

def record_request(path, body, call_index, attempt):
    wire = json.dumps(body).encode()
    audit = copy.deepcopy(body)
    images = []
    for mi, message in enumerate(audit.get('messages', [])):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for ci, part in enumerate(content):
            if part.get('type') != 'image_url':
                continue
            url = part['image_url']['url']
            digest = hashlib.sha256(url.encode()).hexdigest()
            images.append(dict(message_index=mi, content_index=ci, url_sha256=digest, url_length=len(url)))
            part['image_url']['url'] = 'sha256:'+digest
    row = dict(call_index=call_index, attempt=attempt, request_sha256=hashlib.sha256(wire).hexdigest(),
               request_bytes=len(wire), images=images, body=audit,
               scope='Exact request text retained; image URLs represented by SHA256; audit only, not training export')
    with open(path, 'a') as stream:
        stream.write(json.dumps(row)+'\n')
