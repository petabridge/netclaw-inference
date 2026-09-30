#!/usr/bin/env python3
"""Synthetic C1 decode fixture replay; does not execute or grade generated code.

Supply an already running endpoint. No service lifecycle or weight downloads.
This public runner reconstructs the historical core protocol; it is not the
original campaign harness and does not retroactively revalidate its results.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request

PROMPTS = {
    'legacy_coding': 'Implement a production-quality Python async work queue with dataclasses, type hints, bounded concurrency, retries with jitter, structured logging, cancellation safety, and unit tests. Continue until cut off.',
    'legacy_technical_prose': 'Write a precise technical essay explaining how memory bandwidth, arithmetic intensity, and batching interact in modern AI inference. Continue until cut off.',
}


def request(base_url, model, prompt, cap, api_key):
    body={'model':model,'messages':[{'role':'user','content':prompt}], 'temperature':0,
          'max_tokens':cap,'min_tokens':cap,'ignore_eos':True,'stop':[],
          'stream':True,'stream_options':{'include_usage':True},
          'chat_template_kwargs':{'enable_thinking':False}}
    headers={'Content-Type':'application/json'}
    if api_key: headers['Authorization']='Bearer '+api_key
    req=urllib.request.Request(base_url.rstrip('/')+'/chat/completions',json.dumps(body).encode(),headers)
    start=time.monotonic();first=last=None;usage=None;finish=None;done=False
    with urllib.request.urlopen(req,timeout=180) as response:
        for raw in response:
            line=raw.decode().strip()
            if not line.startswith('data:'):continue
            data=line[5:].strip()
            if data=='[DONE]':done=True;break
            event=json.loads(data)
            if event.get('error'):raise RuntimeError('server error in stream')
            if event.get('usage'):usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if delta.get('tool_calls') or delta.get('reasoning_content'):
                    raise RuntimeError('unexpected tool/reasoning output in plain decode fixture')
                if delta.get('content'):
                    now=time.monotonic();first=first if first is not None else now;last=now
                if choice.get('finish_reason'):finish=choice['finish_reason']
    end=time.monotonic()
    if not done or not usage or usage.get('completion_tokens')!=cap or finish!='length' or first is None or last<=first:
        raise RuntimeError('incomplete stream or wrong output length')
    return {'decode_tok_s':(cap-1)/(last-first),'ttft_s':first-start,'e2e_s':end-start,
            'prompt_tokens':usage['prompt_tokens'],'completion_tokens':cap,
            'cached_tokens':(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'finish_reason':finish,'stream_complete':done}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True,help='OpenAI-compatible URL ending in /v1')
    p.add_argument('--model',required=True)
    p.add_argument('--nonce-seed',required=True,help='Use the SAME new seed for baseline/candidate; later repeats may hit cache')
    p.add_argument('--repetitions',type=int,default=5)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.repetitions<1:p.error('positive repetitions required')
    # Exclusive output prevents accidentally overwriting earlier evidence.
    with open(a.output,'x') as out:
        request(a.base_url,a.model,'Count from 1 upward, numbers only.',64,os.getenv('BENCH_API_KEY',''))
        for rep in range(a.repetitions):
            names=list(PROMPTS) if rep%2==0 else list(reversed(PROMPTS))
            for name in names:
                nonce=hashlib.sha256(f'{a.nonce_seed}:{rep}:{name}'.encode()).hexdigest()[:16]
                prompt=f'Synthetic benchmark ID {nonce}.\n{PROMPTS[name]}'
                result=request(a.base_url,a.model,prompt,512,os.getenv('BENCH_API_KEY',''))
                out.write(json.dumps({'workload':name,'rep':rep,**result})+'\n');out.flush()
