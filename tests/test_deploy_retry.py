"""Bounded retries only for safe Vercel reads and content-addressed uploads."""
import io
import ssl
import unittest
from unittest.mock import patch
import deploy as deployment

class SafeTransportRetryTests(unittest.TestCase):
    def response(self):
        from unittest.mock import MagicMock
        response = MagicMock(); response.status = 200; response.read.return_value = b'{}'; response.__enter__.return_value = response
        return response

    def reset(self):
        return deployment.urllib.error.URLError(ConnectionResetError(104, 'fixture connection reset'))

    def test_content_addressed_upload_recovers_from_connection_reset(self):
        with patch.object(deployment.urllib.request, 'urlopen', side_effect=[self.reset(), self.response()]) as request, patch('time.sleep') as wait:
            self.assertEqual(deployment.req('https://api.vercel.com/v2/files?teamId=fixture', data=b'public bytes', method='POST'), (200,b'{}'))
            self.assertEqual(request.call_count, 2); wait.assert_called_once()

    def test_vercel_metadata_read_recovers_from_connection_reset(self):
        with patch.object(deployment.urllib.request, 'urlopen', side_effect=[self.reset(), self.response()]) as request, patch('time.sleep'):
            self.assertEqual(deployment.req('https://api.vercel.com/v9/projects/fixture'), (200,b'{}'))
            self.assertEqual(request.call_count, 2)

    def test_transient_retries_are_bounded(self):
        with patch.object(deployment.urllib.request, 'urlopen', side_effect=self.reset()) as request, patch('time.sleep') as wait:
            with self.assertRaises(deployment.urllib.error.URLError):
                deployment.req('https://api.vercel.com/v2/files', data=b'public', method='POST')
            self.assertEqual(request.call_count, 3); self.assertEqual(wait.call_count, 2)

    def test_uncertain_deployment_creation_is_never_repeated(self):
        with patch.object(deployment.urllib.request, 'urlopen', side_effect=self.reset()) as request, patch('time.sleep') as wait:
            with self.assertRaises(deployment.urllib.error.URLError):
                deployment.req('https://api.vercel.com/v13/deployments', data=b'{}', method='POST')
            self.assertEqual(request.call_count, 1); wait.assert_not_called()

    def test_http_denials_and_rate_limits_are_returned_without_retry(self):
        import io
        for status in [401,403,404,429]:
            with self.subTest(status=status):
                error = deployment.urllib.error.HTTPError('https://api.vercel.com/v2/files', status, 'fixture', {}, io.BytesIO(b'fixture'))
                with patch.object(deployment.urllib.request, 'urlopen', side_effect=error) as request, patch('time.sleep') as wait:
                    self.assertEqual(deployment.req('https://api.vercel.com/v2/files',data=b'public',method='POST'), (status,b'fixture'))
                    self.assertEqual(request.call_count,1); wait.assert_not_called()

    def test_certificate_failure_is_not_retried(self):
        import ssl
        with patch.object(deployment.urllib.request, 'urlopen', side_effect=deployment.urllib.error.URLError(ssl.SSLCertVerificationError(1,'fixture'))) as request, patch('time.sleep') as wait:
            with self.assertRaises(deployment.urllib.error.URLError): deployment.req('https://api.vercel.com/v9/projects/fixture')
            self.assertEqual(request.call_count,1); wait.assert_not_called()

    def test_other_origins_are_not_retried(self):
        with patch.object(deployment.urllib.request, 'urlopen', side_effect=self.reset()) as request, patch('time.sleep') as wait:
            with self.assertRaises(deployment.urllib.error.URLError): deployment.req('https://example.invalid/v2/files',data=b'public',method='POST')
            self.assertEqual(request.call_count,1); wait.assert_not_called()



class ConnectionTimeoutRegressionTests(unittest.TestCase):
    def test_connect_timeout_on_file_upload_is_bounded_and_can_recover(self):
        from unittest.mock import MagicMock
        response = MagicMock(); response.status = 200; response.read.return_value = b'ok'; response.__enter__.return_value = response
        with patch.object(deployment.urllib.request, 'urlopen', side_effect=[deployment.urllib.error.URLError(TimeoutError('connect timed out')), response]) as request, patch('time.sleep') as wait:
            self.assertEqual(deployment.req('https://api.vercel.com/v2/files', data=b'content-addressed', headers={'x-vercel-digest':'fixture'}, method='POST'), (200,b'ok'))
            self.assertEqual(request.call_count, 2); wait.assert_called_once_with(1)

    def test_implicit_post_creation_and_other_writes_never_retry(self):
        for url, method in [('https://api.vercel.com/v13/deployments', None), ('https://api.vercel.com/v9/projects/fixture', 'PATCH')]:
            with self.subTest(url=url), patch.object(deployment.urllib.request, 'urlopen', side_effect=TimeoutError('uncertain')) as request, patch('time.sleep') as wait:
                with self.assertRaises(TimeoutError): deployment.req(url, data=b'{}', method=method)
                self.assertEqual(request.call_count, 1); wait.assert_not_called()

    def test_all_http_failures_return_without_retry(self):
        for status in [400,401,403,404,409,429,500,502,503,504]:
            failure = deployment.urllib.error.HTTPError('https://api.vercel.com/v2/files', status, 'fixture', {'Retry-After':'5'}, io.BytesIO(b'failed'))
            with self.subTest(status=status), patch.object(deployment.urllib.request, 'urlopen', side_effect=failure) as request, patch('time.sleep') as wait:
                self.assertEqual(deployment.req('https://api.vercel.com/v2/files', data=b'public', method='POST'),(status,b'failed'))
                self.assertEqual(request.call_count,1); wait.assert_not_called()

if __name__ == '__main__': unittest.main()
