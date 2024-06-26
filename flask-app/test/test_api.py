import unittest
import json




class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True


    def test_detect_valid_data(self):
        # valid test data
        valid_data = [
            {
                "user_id": 1,
                "signup_time": "2023-01-01 00:00:00",
                "purchase_time": "2023-01-01 01:00:00",
                "purchase_value": 100.0,
                "device_id": "device_1",
                "source": "SEO",
                "browser": "Chrome",
                "sex": "M",
                "age": 30,
                "ip_address": 732758368,
                "country": "US"
            },
        ]

        response = self.app.post('/detect', data=json.dumps(valid_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Detection', json.loads(response.data))

    def test_detect_invalid_data(self):
        # Missing purchase_time key
        invalid_data = [
            {
                "user_id": 1,
                "signup_time": "2023-01-01 00:00:00",
                "purchase_value": 100.0,
                "device_id": "device_1",
                "source": "SEO",
                "browser": "Chrome",
                "sex": "M",
                "age": 30,
                "ip_address": 732758368,
                "country": "US"
            }
        ]

        response = self.app.post('/detect', data=json.dumps(invalid_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', json.loads(response.data))

if __name__ == '__main__':
    unittest.main()
