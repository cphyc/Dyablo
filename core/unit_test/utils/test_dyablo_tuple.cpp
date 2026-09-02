#include "gtest/gtest.h"

#include "utils/misc/dyablo_tuple.h"

using namespace dyablo;

TEST(Test_dyablo_tuple, make_get_match)
{
  auto t = make_dyablo_tuple( 1.0, 2.0 );

  EXPECT_EQ( 1.0, dyablo_tuple_get<0>(t) );
  EXPECT_EQ( 2.0, dyablo_tuple_get<1>(t) );
}

TEST(Test_dyablo_tuple, tie_get_match)
{
  double a = 1.0, b = 2.0;
  auto t = dyablo_tuple_tie( a, b );

  EXPECT_EQ( 1.0, dyablo_tuple_get<0>(t) );
  EXPECT_EQ( 2.0, dyablo_tuple_get<1>(t) );
}

TEST(Test_dyablo_tuple, tie_assign_match)
{
  double a = 1.0, b = 2.0;
  auto t = dyablo_tuple_tie( a, b );

  dyablo_tuple_get<0>(t) = 3.0;
  dyablo_tuple_get<1>(t) = 4.0;

  EXPECT_EQ( 3.0, a );
  EXPECT_EQ( 4.0, b );
}

TEST(Test_dyablo_tuple, ref_tie_assign)
{
  double a, b;
  dyablo_tuple_tie( a, b ) = make_dyablo_tuple(1.0, 2.0);

  EXPECT_EQ( 1.0, a );
  EXPECT_EQ( 2.0, b );
}

TEST(Test_dyablo_tuple, structured_binding_assign)
{
  double a = 1.0, b = 2.0;
  auto [x, y] = dyablo_tuple_tie(a,b);

  EXPECT_EQ( 1.0, x );
  EXPECT_EQ( 2.0, y );

  x = 3.0;
  y = 4.0;

  EXPECT_EQ( 3.0, a );
  EXPECT_EQ( 4.0, b );
}

TEST(Test_dyablo_tuple, structured_binding_assign_array)
{
  double a = 1.0, b[3] = {2.0,3.0,4.0};
  auto [x, y] = dyablo_tuple_tie(a,b);

  EXPECT_EQ( 1.0, x );
  EXPECT_EQ( 2.0, y[0] );
  EXPECT_EQ( 3.0, y[1] );
  EXPECT_EQ( 4.0, y[2] );

  x = 5.0;
  y[0] = 6.0;
  y[1] = 7.0;
  y[2] = 8.0;

  EXPECT_EQ( 5.0, a );
  EXPECT_EQ( 6.0, b[0] );
  EXPECT_EQ( 7.0, b[1] );
  EXPECT_EQ( 8.0, b[2] );
}