import 'dart:js' as js;
import 'dart:convert';

import 'package:chrome_extension/chrome.dart';
import 'package:chrome_extension/tabs.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:http/http.dart' as http;
import 'package:html/parser.dart';
import 'package:nlp_chrome_extension_flutter/enums.dart';
import 'package:nlp_chrome_extension_flutter/font.dart';

import 'chrome_api.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NLP Project Chrome Extension',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFC75C5C),
        ),
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  /// 내부 로직 실행을 위한 값들
  String? url;
  String? title;
  String? contents;

  /// 최종 결과값
  bool? isFake;
  String? fixedTitle;

  /// UI 관련
  bool isLoading = false;
  PageState state = PageState.start;
  String? message;

  Future<void> _clickButton() async {
    setState(() {
      isLoading = true;
    });

    final tabs = await chrome.tabs.query(
      QueryInfo(
          active: true,
          currentWindow: true
      ),
    );

    if(tabs.isEmpty || tabs.first.id == null) {
      setState(() {
        isLoading = false;
        state = PageState.error;
        message = 'Cannot find active tab :(';
      });
      return;
    }

    url = tabs.first.url;
    if(url == null) {
      setState(() {
        isLoading = false;
        state = PageState.error;
        message = 'Cannot find url of current tab :(';
      });
      return;
    }
    else if(url!.startsWith('https://n.news.naver.com/article/') == false) {
      setState(() {
        isLoading = false;
        state = PageState.error;
        message = 'This extension only works on the Naver news article site.\n'
            'Please check if the address starts with \'https://n.news.naver.com/article/\'.';
      });
      return;
    }

    js.context.callMethod('getHtmlCode');
    await Future.delayed(const Duration(microseconds: 2000));
    await chrome.storage.local.get('data');
    final htmlCode = (await chrome.storage.local.get('data'))['data'];
    final document = parse(htmlCode);

    final titleContent = document.querySelector(
      '.n_news > .end_container > #ct_wrap > .ct_scroll_wrapper > .newsct_wrapper > #ct > .media_end_head > .media_end_head_title',
    );
    final newsContent = document.querySelectorAll(
      '.n_news > .end_container > #ct_wrap > .ct_scroll_wrapper > .newsct_wrapper > #ct > #contents > #newsct_article > .go_trans',
    );

    if(titleContent == null || newsContent.isEmpty) {
      setState(() {
        isLoading = false;
        state = PageState.error;
        message = 'Cannot find title and contents of this news :(';
      });
      return;
    }

    final newsTitle = titleContent.text;
    String newsTexts = newsContent.first.text;

    final summary = newsContent.first.querySelector('.media_end_summary');
    final imgDescription = newsContent.first.querySelector('.end_photo_org > .img_desc');
    if(summary != null) {
      newsTexts = newsTexts.replaceFirst(summary.text, '');
    }
    if(imgDescription != null) {
      newsTexts = newsTexts.replaceFirst(imgDescription.text, '');
    }
    newsTexts = newsTexts.trim(); // 문자열 앞/뒤의 whitespace 전부 제거

    setState(() {
      isLoading = false;
      state = PageState.notFake;
      title = newsTitle;
      contents = newsTexts;
      message = null;
    });

    try {
      final aiResponse = await http.post(
          Uri.parse('http://127.0.0.1:8000/scrape'),
          headers: {
            "Access-Control-Allow-Origin": "*",
            'Content-Type': 'application/json',
            'Accept': '*/*'
          },
          body: {
            "title": newsTitle,
            "content": newsTexts,
          }
      );

      if(aiResponse.statusCode == 200) {
        final result = json.decode(aiResponse.body);

        // 응답 형태 : {"is_reliable": result[0], "new_title": result[1]}
        if(result is Map<String, dynamic> && result.containsKey('is_reliable') && result.containsKey('new_title')) {
          if(result['is_reliable'] != 'clickbait') {
            setState(() {
              isLoading = false;
              state = PageState.notFake;
              title = newsTitle;
              contents = newsTexts;
              message = null;
            });
          } else {
            setState(() {
              isLoading = false;
              state = PageState.fakeShowTitles;
              title = newsTitle;
              fixedTitle = result['new_title'];
              contents = newsTexts;
              message = null;
            });
          }
        } else {
          setState(() {
            isLoading = false;
            state = PageState.error;
            message = "오류가 발생했습니다 : 응답 변환에 실패했습니다.\n   (${aiResponse.body})";
          });
        }
      } else {
        setState(() {
          isLoading = false;
          state = PageState.error;
          message = "오류가 발생했습니다 : ${aiResponse.statusCode}\n${aiResponse.body}";
        });
      }
    } on Exception catch(e) {
      setState(() {
        isLoading = false;
        state = PageState.error;
        message = "오류가 발생했습니다 : $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: Text(
          'NLP Project Chrome Extension',
          style: AppTextStyle.appbar,
        ),
        elevation: 1,
      ),
      body: isLoading
          ? const Center(
        child: CircularProgressIndicator(),
      ) : state == PageState.start
          ? startPage()
          : state == PageState.notFake
          ? notFakePage()
          : state == PageState.fakeShowTitles
          ? fakeShowTitlesPage()
          : state == PageState.fakeWithContents
          ? fakeWithContentsPage()
          : errorPage(),
    );
  }

  Widget startPage() {
    return Center(
      child: _mainButton(
        'Is the title of this article fake?',
      ),
    );
  }

  Widget errorPage() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Spacer(),
          SizedBox(
            width: 318,
            child: Text(
              message ?? '',
              style: AppTextStyle.articleTitle,
            ),
          ),
          const Spacer(),
          _mainButton(
            'Reload another article',
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget notFakePage() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(height: 8),
          Text.rich(
            TextSpan(
              text: 'The title of below is ',
              style: AppTextStyle.resultTitle,
              children: [
                TextSpan(
                  text: 'NOT clickbait!!!',
                  style: AppTextStyle.resultTitle.copyWith(
                    color: const Color(0xFFFD9795),
                    decoration: TextDecoration.underline,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Container(
            width: 318,
            height: 1,
            color: const Color(0xFFDCDEE5),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: 318,
            child: Text(
              'Title : $title',
              style: AppTextStyle.articleTitle,
            ),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: SingleChildScrollView(
              child: SizedBox(
                width: 318,
                child: Text(
                  contents ?? '',
                  style: AppTextStyle.articleContents,
                ),
              ),
            ),
          ),
          _mainButton(
            'Reload another article',
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget fakeShowTitlesPage() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(height: 8),
          Text.rich(
            TextSpan(
              text: 'The title of below is ',
              style: AppTextStyle.resultTitle,
              children: [
                TextSpan(
                  text: 'clickbait!!!',
                  style: AppTextStyle.resultTitle.copyWith(
                    color: const Color(0xFFFD9795),
                    decoration: TextDecoration.underline,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Container(
            width: 318,
            height: 1,
            color: const Color(0xFFDCDEE5),
          ),
          const SizedBox(height: 20),

          SizedBox(
            width: 318,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Original title',
                  style: AppTextStyle.fakeTitlePagesubTitle.copyWith(
                    color: const Color(0xFF848282),
                  ),
                ),
                Text(
                  title!,
                  style: AppTextStyle.fakeTitle,
                  textAlign: TextAlign.start,
                ),
                const SizedBox(height: 20),
                Container(
                  alignment: Alignment.center,
                  child: SvgPicture.asset(
                    'assets/arrow.svg',
                    width: 40,
                    height: 102,
                  ),
                ),
                const SizedBox(height: 20),
                Text(
                  'Fixed title',
                  style: AppTextStyle.fakeTitlePagesubTitle.copyWith(
                    color: const Color(0xFF141414),
                  ),
                ),
                Text(
                  fixedTitle ?? '',
                  style: AppTextStyle.articleTitle,
                  textAlign: TextAlign.start,
                ),
              ],
            ),
          ),
          const Spacer(),
          Container(
            margin: const EdgeInsets.only(right: 12),
            alignment: Alignment.centerRight,
            child: InkWell(
              onTap: () {
                setState(() {
                  isLoading = false;
                  state = PageState.fakeWithContents;
                });
              },
              child: Container(
                width: 195,
                height: 32,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: const Color(0xFFCBC9C9),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'Show with contents',
                      style: AppTextStyle.showWithContentsButton,
                    ),
                    SvgPicture.asset(
                      'assets/ic.svg',
                      width: 24,
                      height: 24,
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
          _mainButton(
            'Reload another article',
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget fakeWithContentsPage() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(height: 8),
          Text.rich(
            TextSpan(
              text: 'The title of below is ',
              style: AppTextStyle.resultTitle,
              children: [
                TextSpan(
                  text: 'clickbait!!!',
                  style: AppTextStyle.resultTitle.copyWith(
                    color: const Color(0xFFFD9795),
                    decoration: TextDecoration.underline,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          Container(
            width: 318,
            height: 1,
            color: const Color(0xFFDCDEE5),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: 318,
            child: Text(
              'Title : $fixedTitle',
              style: AppTextStyle.articleTitle,
            ),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: SingleChildScrollView(
              child: SizedBox(
                width: 318,
                child: Text(
                  contents ?? '',
                  style: AppTextStyle.articleContents,
                ),
              ),
            ),
          ),
          _mainButton(
            'Reload another article',
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget _mainButton(String text) {
    return InkWell(
      onTap: _clickButton,
      child: Container(
        width: 326,
        height: 56,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: const Color(0xFFFD9795),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          text,
          style: AppTextStyle.mainButton,
        ),
      ),
    );
  }
}
