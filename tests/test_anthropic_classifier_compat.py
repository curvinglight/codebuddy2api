#!/usr/bin/env python3
"""
test_anthropic_classifier_compat.py — /v1/messages 与 Claude Code auto 模式分类器的接口兼容性。

分类器（side query）按 Anthropic 协议发请求时不带 stream 字段，因此必须拿到 JSON
Message；拿到 SSE 时 SDK 会返回一个 stop_reason / usage / content 全为 undefined 的
对象，读 usage.input_tokens 直接抛错，分类器表现为「模型暂时不可用」。
"""

import asyncio
import json

import httpx
import pytest

from core import converter

# 必须在任何 monkeypatch 之前抓住原始 client，converter.httpx 与本模块的 httpx 是同一个模块对象。
_REAL_CLIENT = httpx.AsyncClient

CHAT_SSE = (
    b'data: {"choices":[{"delta":{"content":"<block>no</block>"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":50,"completion_tokens":7,"total_tokens":57}}\n\n'
    b'data: [DONE]\n\n'
)

CHAT_SSE_NO_USAGE = (
    b'data: {"choices":[{"delta":{"content":"<block>no</block>"},"finish_reason":"stop"}]}\n\n'
    b'data: [DONE]\n\n'
)


def _post(payload):
    async def run():
        async with _REAL_CLIENT(
            transport=httpx.ASGITransport(app=converter.app), base_url='http://test'
        ) as client:
            return await client.post('/v1/messages', json=payload)

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(converter, '_check_auth', lambda *a: None)
    monkeypatch.setattr(converter, '_cred', lambda: type('Credential', (), {'get_headers': lambda self: {}})())
    monkeypatch.setattr(converter, '_log', lambda *a: None)
    monkeypatch.setitem(converter.CONFIG, 'desensitize', False)
    _patch_upstream(monkeypatch, CHAT_SSE)


def _patch_upstream(monkeypatch, body):
    def upstream(request):
        return httpx.Response(200, content=body)

    monkeypatch.setattr(
        converter.httpx,
        'AsyncClient',
        lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(upstream), **kw),
    )


def test_messages_without_stream_returns_json():
    """不带 stream 的请求（分类器的形态）必须返回 JSON Message 而不是 SSE。"""
    resp = _post({'model': 'deepseek-v4.1-flash', 'max_tokens': 32,
                  'messages': [{'role': 'user', 'content': 'hi'}]})

    assert resp.status_code == 200
    assert resp.headers['content-type'].startswith('application/json')
    msg = resp.json()
    assert msg['type'] == 'message'
    assert msg['stop_reason'] == 'end_turn'
    assert msg['content'][0]['text'] == '<block>no</block>'


def test_classifier_consumes_usage_and_stop_reason():
    """模拟 Claude Code 分类器的读取方式：stop_reason 与 usage.input_tokens 都必须可读。"""
    resp = _post({'model': 'deepseek-v4.1-flash', 'max_tokens': 32,
                  'messages': [{'role': 'user', 'content': 'hi'}]})
    msg = resp.json()

    assert msg['stop_reason'] in ('end_turn', 'stop_sequence')
    assert msg['usage']['input_tokens'] == 50
    assert msg['usage']['output_tokens'] == 7


def test_usage_present_when_backend_omits_it(monkeypatch):
    """后端没回 usage 时也不能把字段省掉，否则调用方读 usage.input_tokens 抛错。"""
    _patch_upstream(monkeypatch, CHAT_SSE_NO_USAGE)
    resp = _post({'model': 'deepseek-v4.1-flash', 'max_tokens': 32,
                  'messages': [{'role': 'user', 'content': 'hi'}]})

    assert resp.json()['usage'] == {'input_tokens': 0, 'output_tokens': 0}


def test_stream_true_still_returns_sse():
    """显式 stream=true 保持流式行为不变。"""
    resp = _post({'model': 'deepseek-v4.1-flash', 'max_tokens': 32, 'stream': True,
                  'messages': [{'role': 'user', 'content': 'hi'}]})

    assert resp.headers['content-type'].startswith('text/event-stream')
    assert 'message_stop' in resp.text


def test_stop_sequences_forwarded_to_backend(monkeypatch):
    """Anthropic 的 stop_sequences 要映射成 Chat 的 stop，分类器靠它截断判定标签。"""
    captured = []

    def upstream(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=CHAT_SSE)

    monkeypatch.setattr(
        converter.httpx,
        'AsyncClient',
        lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(upstream), **kw),
    )
    _post({'model': 'm', 'max_tokens': 32, 'stop_sequences': ['</block>'],
           'messages': [{'role': 'user', 'content': 'hi'}]})
    assert captured[0]['stop'] == ['</block>']
