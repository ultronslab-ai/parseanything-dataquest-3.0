import urllib.request
import json
import uuid
import sys

def upload_and_check(filepath, label):
    url = "http://127.0.0.1:8585/api/v1/documents/upload"
    with open(filepath, "rb") as f:
        pdf_bytes = f.read()

    boundary = "----TestBoundary" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{label}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + pdf_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        doc_id = data.get("document_id")
        blocks = data.get("blocks", [])
        equations = data.get("stats", {}).get("equations_detected", 0)
        conf = data.get("stats", {}).get("average_confidence")
        print(f"\n=== Test {label} ===")
        print(f"Status: {resp.status} | Doc ID: {doc_id}")
        print(f"File Type: {data.get('file_type')} | Blocks: {len(blocks)} | Equations: {equations}")
        print(f"Extraction Method of Block 1: {blocks[0].get('extraction_method') if blocks else 'N/A'}")
        
        # Test structured-json endpoint
        sj_url = f"http://127.0.0.1:8585/api/v1/documents/{doc_id}/structured-json"
        with urllib.request.urlopen(sj_url) as sj_resp:
            sj = json.loads(sj_resp.read().decode("utf-8"))
            print(f"Structured JSON: OK ({sj_resp.status})")
            print(f"Statistics: {sj.get('statistics')}")
            if sj.get("pages"):
                print(f"Page Type: {sj['pages'][0].get('page_type')}")

if __name__ == "__main__":
    upload_and_check("backend/data/test_digital_sample.pdf", "test_digital_sample.pdf")
    upload_and_check("backend/data/test_emt_p1.pdf", "test_emt_p1.pdf")
