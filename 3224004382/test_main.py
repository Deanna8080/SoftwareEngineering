import unittest
import tempfile
import os
from main import cosine_similarity, normalize, ngram_counter, read_file


class TestPlagiarism(unittest.TestCase):

    def test_identical_text(self):
        # 测试1：完全相同的文本，相似度应为 1.0
        score = cosine_similarity("今天是星期天", "今天是星期天")
        self.assertAlmostEqual(score, 1.0)

    def test_sample(self):
        # 测试2：作业样例，相似度应大于 0.5 且小于 1.0
        score = cosine_similarity(
            "今天是星期天，天气晴，今天晚上我要去看电影。",
            "今天是周天，天气晴朗，我晚上要去看电影。"
        )
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)

    def test_empty_original(self):
        # 测试3：原文为空，相似度应为 0.0
        self.assertEqual(cosine_similarity("", "任意文本"), 0.0)

    def test_empty_copy(self):
        # 测试4：抄袭版为空，相似度应为 0.0
        self.assertEqual(cosine_similarity("任意文本", ""), 0.0)

    def test_punctuation_ignored(self):
        # 测试5：标点符号、空格应被忽略，相似度应为 1.0
        score = cosine_similarity("今天，天气 晴！", "今天天气晴")
        self.assertAlmostEqual(score, 1.0)

    def test_english_case_ignored(self):
        # 测试6：英文大小写应被忽略，相似度应为 1.0
        score = cosine_similarity("Hello World", "hello world")
        self.assertAlmostEqual(score, 1.0)

    def test_short_text(self):
        # 测试7：极短文本（长度小于 n），应能处理
        score = cosine_similarity("我", "我")
        self.assertAlmostEqual(score, 1.0)

    def test_different_text(self):
        # 测试8：完全不同的文本，相似度应小于 0.2
        score = cosine_similarity("苹果香蕉橘子", "计算机程序设计")
        self.assertLess(score, 0.2)

    def test_ngram_counter(self):
        # 测试9：测试 n-gram 切分是否正确
        counter = ngram_counter("今天天", 2)
        self.assertEqual(counter["今天"], 1)
        self.assertEqual(counter["天天"], 1)

    def test_invalid_n(self):
        # 测试10：非法 n 值应抛出 ValueError 异常
        with self.assertRaises(ValueError):
            ngram_counter("abc", 0)

    def test_read_file(self):
        # 测试11：测试文件读取功能
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("测试文本")
            path = f.name

        try:
            content = read_file(path)
            self.assertEqual(content, "测试文本")
        finally:
            os.remove(path)

    def test_long_text_sample(self):
        """
        集成测试：使用老师下发的长文本测试样例
        验证程序在真实长文本下的表现，要求相似度在合理区间内
        """
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        orig_path = os.path.join(base_dir, "samples", "orig.txt")
        copy_path = os.path.join(base_dir, "samples", "orig_0.8_add.txt")

        if not os.path.exists(orig_path) or not os.path.exists(copy_path):
            self.skipTest("长文本样例未下载，跳过集成测试")

        original_text = read_file(orig_path)
        copied_text = read_file(copy_path)
        score = cosine_similarity(original_text, copied_text, n=2)

        self.assertGreater(score, 0.7)
        self.assertLess(score, 0.99)
        print(f"\n长文本样例相似度: {score:.4f}")

if __name__ == "__main__":
    unittest.main()