import 'package:flutter/material.dart';

abstract class AppTextStyle {
  AppTextStyle._();

  static TextStyle appbar = PretendardTextStyle.semiBold(
    size: 16,
    height: 20,
    letterSpacing: -0.6,
  );

  static TextStyle mainButton = PretendardTextStyle.semiBold(
    size: 16,
    height: 20,
    letterSpacing: -0.6,
    color: const Color(0xFFFFFFFF),
  );

  static TextStyle resultTitle = PretendardTextStyle.bold(
    size: 20,
    height: 30,
    letterSpacing: 0.4,
  );

  static TextStyle articleTitle = PretendardTextStyle.medium(
    size: 16,
    height: 22,
    letterSpacing: -0.1,
  );

  static TextStyle articleContents = PretendardTextStyle.medium(
    size: 13,
    height: 22,
    letterSpacing: -0.6,
    color: const Color(0xFF4A4D57),
  );

  static TextStyle fakeTitle = PretendardTextStyle.medium(
    size: 15,
    height: 22,
    letterSpacing: -0.1,
    color: const Color(0xFF848282),
    decoration: TextDecoration.lineThrough,
  );

  static TextStyle fakeTitlePagesubTitle = PretendardTextStyle.medium(
    size: 14,
    height: 22,
    letterSpacing: -0.1,
    decoration: TextDecoration.underline,
  );

  static TextStyle showWithContentsButton = PretendardTextStyle.bold(
    size: 14,
    height: 20,
    letterSpacing: 0,
    color: const Color(0xFF242424),
  );
}

/// 새로운 Design Guide
@immutable
class PretendardTextStyle extends TextStyle {
  static const pretendardBold = 'pretendard_bold';
  static const pretendardRegular = 'pretendard_regular';
  static const pretendardSemiBold = 'pretendard_semiBold';
  static const pretendardMedium = 'pretendard_medium';

  static const _black = Color(0xFF141414);

  const PretendardTextStyle(
    String fontFamily,
    Color color,
    double size,
    FontWeight fontWeight,
    double height,
    double? letterSpacing,
    TextDecoration? decoration,
  ) : super(
    fontFamily: fontFamily,
    color: color,
    fontSize: size,
    fontWeight: fontWeight,
    height: height / size,
    letterSpacing: letterSpacing,
    leadingDistribution: TextLeadingDistribution.even,
    decoration: decoration,
  );

  factory PretendardTextStyle.regular({
    required double size,
    Color color = _black,
    FontWeight fontWeight = FontWeight.normal,
    double height = 1.0,
    double? letterSpacing,
    TextDecoration? decoration,
  }) =>
      PretendardTextStyle(
          pretendardRegular, color, size, fontWeight, height, letterSpacing, decoration);

  factory PretendardTextStyle.semiBold({
    required double size,
    Color color = _black,
    FontWeight fontWeight = FontWeight.normal,
    double height = 1.0,
    double? letterSpacing,
    TextDecoration? decoration,
  }) =>
      PretendardTextStyle(
          pretendardSemiBold, color, size, fontWeight, height, letterSpacing, decoration);

  factory PretendardTextStyle.medium({
    required double size,
    Color color = _black,
    FontWeight fontWeight = FontWeight.normal,
    double height = 1.0,
    double? letterSpacing,
    TextDecoration? decoration,
  }) =>
      PretendardTextStyle(
          pretendardMedium, color, size, fontWeight, height, letterSpacing, decoration);

  factory PretendardTextStyle.bold({
    required double size,
    Color color = _black,
    FontWeight fontWeight = FontWeight.normal,
    double height = 1.0,
    double? letterSpacing,
    TextDecoration? decoration,
  }) =>
      PretendardTextStyle(
          pretendardBold, color, size, fontWeight, height, letterSpacing, decoration);
}
