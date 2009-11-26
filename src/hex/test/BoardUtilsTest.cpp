//---------------------------------------------------------------------------
/** @file BoardUtilsTest.cpp
 */
//---------------------------------------------------------------------------
#include <boost/test/auto_unit_test.hpp>

#include "BoardUtils.hpp"
#include "HexBoard.hpp"

using namespace benzene;

//---------------------------------------------------------------------------

namespace {

BOOST_AUTO_TEST_CASE(BoardUtils_BitsetPacking)
{
    BOOST_REQUIRE(MAX_WIDTH >= 7 && MAX_HEIGHT >= 9);
    ConstBoard* cb = &ConstBoard::Get(7, 9);
    bitset_t b1, b2;
    b2 = BoardUtils::PackBitset(*cb, b1);
    BOOST_CHECK_EQUAL(BoardUtils::UnpackBitset(*cb, b2), b1);
    b1.flip();
    b2 = BoardUtils::PackBitset(*cb, b1);
    BOOST_CHECK_EQUAL(BoardUtils::UnpackBitset(*cb, b2), b1 & cb->GetCells());
    BOOST_CHECK_EQUAL(b1.count(), (std::size_t)BITSETSIZE);
    BOOST_CHECK_EQUAL(b2.count(), cb->GetCells().count());
    b1.reset();
    b1.set(SWAP_PIECES);
    b1.set(NORTH);
    b1.set(FIRST_CELL);
    int adjustment = 1;
    if (FIRST_INVALID != BITSETSIZE) {
	b1.set(FIRST_INVALID);
	adjustment = 0;
    }
    b2 = BoardUtils::PackBitset(*cb, b1);
    BOOST_CHECK_EQUAL(b1.count(), (std::size_t)(4 - adjustment));
    BOOST_CHECK_EQUAL(b2.count(), 1u);
    BOOST_CHECK_EQUAL(BoardUtils::UnpackBitset(*cb, b2), b1 & cb->GetCells());
}

BOOST_AUTO_TEST_CASE(BoardUtils_RotateAndMirror)
{
    BOOST_REQUIRE(MAX_WIDTH >= 11 && MAX_HEIGHT >= 11);
    
    // rotating edges
    ConstBoard* cb = &ConstBoard::Get(11, 11);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, NORTH), SOUTH);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, EAST), WEST);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, BoardUtils::Rotate(*cb, EAST)), EAST);
    
    // mirroring edges
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, NORTH), WEST);
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, EAST), SOUTH);
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, BoardUtils::Mirror(*cb, WEST)), WEST);
    
    // rotation of points on board
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_F6), HEX_CELL_F6);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_A1), HEX_CELL_K11);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_B1), HEX_CELL_J11);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_A2), HEX_CELL_K10);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_D9), HEX_CELL_H3);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_H3), HEX_CELL_D9);
    
    // mirroring points on board
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, HEX_CELL_F6), HEX_CELL_F6);
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, HEX_CELL_A1), HEX_CELL_A1);
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, HEX_CELL_B1), HEX_CELL_A2);
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, HEX_CELL_A2), HEX_CELL_B1);
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, HEX_CELL_D9), HEX_CELL_I4);
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, HEX_CELL_H3), HEX_CELL_C8);
    
    // rotation of points on rectangular board
    cb = &ConstBoard::Get(9, 6);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_A1), HEX_CELL_I6);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_A3), HEX_CELL_I4);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_E3), HEX_CELL_E4);
    
    // rotation of points on board of even dimensions
    cb = &ConstBoard::Get(8, 8);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_D4), HEX_CELL_E5);
    BOOST_CHECK_EQUAL(BoardUtils::Rotate(*cb, HEX_CELL_D5), HEX_CELL_E4);
    
    // mirroring points on board of even dimensions
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, HEX_CELL_D4), HEX_CELL_D4);
    BOOST_CHECK_EQUAL(BoardUtils::Mirror(*cb, HEX_CELL_D5), HEX_CELL_E4);
}

BOOST_AUTO_TEST_CASE(BoardUtils_CentrePoints)
{
    BOOST_REQUIRE(MAX_WIDTH >= 10 && MAX_HEIGHT >= 10);
    
    // centre points on odd dimension boards
    ConstBoard* cb = &ConstBoard::Get(9, 9);
    BOOST_CHECK_EQUAL(BoardUtils::CenterPoint(*cb), HEX_CELL_E5);
    BOOST_CHECK_EQUAL(BoardUtils::CenterPoint(*cb), BoardUtils::CenterPointRight(*cb));
    BOOST_CHECK_EQUAL(BoardUtils::CenterPoint(*cb), BoardUtils::CenterPointLeft(*cb));
    
    // centre points on even dimension boards
    cb = &ConstBoard::Get(10, 10);
    BOOST_CHECK_EQUAL(BoardUtils::CenterPointLeft(*cb), HEX_CELL_E6);
    BOOST_CHECK_EQUAL(BoardUtils::CenterPointRight(*cb), HEX_CELL_F5);
    
    // centre points on rectangular boards
    cb = &ConstBoard::Get(7, 10);
    BOOST_CHECK_EQUAL(BoardUtils::CenterPointLeft(*cb), HEX_CELL_D5);
    BOOST_CHECK_EQUAL(BoardUtils::CenterPointRight(*cb), HEX_CELL_D6);
    
    cb = &ConstBoard::Get(10, 7);
    BOOST_CHECK_EQUAL(BoardUtils::CenterPointLeft(*cb), HEX_CELL_E4);
    BOOST_CHECK_EQUAL(BoardUtils::CenterPointRight(*cb), HEX_CELL_F4);
}

BOOST_AUTO_TEST_CASE(BoardUtils_CoordsToPoint)
{
    BOOST_REQUIRE(MAX_WIDTH >= 8 && MAX_HEIGHT >= 8);
    ConstBoard* cb = &ConstBoard::Get(8, 8);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, -2, 0), INVALID_POINT);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, 0, -2), INVALID_POINT);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, -1, -1), INVALID_POINT);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, cb->Width(), cb->Height()), INVALID_POINT);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, -1, cb->Height()), INVALID_POINT);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, cb->Width(), -1), INVALID_POINT);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, 0, -1), NORTH);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, -1, 0), WEST);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, -1, cb->Height()-1), WEST);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, cb->Width()-1, cb->Height()), SOUTH);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, cb->Width(), cb->Height()-1), EAST);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, 0, 0), FIRST_CELL);
    BOOST_CHECK_EQUAL(BoardUtils::CoordsToPoint(*cb, cb->Width()-1, cb->Height()-1),
		      HEX_CELL_H8);
}

BOOST_AUTO_TEST_CASE(BoardUtils_PointInDir)
{
    BOOST_REQUIRE(MAX_WIDTH >= 8 && MAX_HEIGHT >= 8);
    ConstBoard* cb = &ConstBoard::Get(8, 8);

    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_B2, DIR_EAST), HEX_CELL_C2);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_B2, DIR_NORTH_EAST), HEX_CELL_C1);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_B2, DIR_NORTH), HEX_CELL_B1);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_B2, DIR_WEST), HEX_CELL_A2);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_B2, DIR_SOUTH_WEST), HEX_CELL_A3);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_B2, DIR_SOUTH), HEX_CELL_B3);

    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_A1, DIR_NORTH_EAST), NORTH);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_A1, DIR_NORTH), NORTH);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_A1, DIR_WEST), WEST);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, HEX_CELL_A1, DIR_SOUTH_WEST), WEST);
    
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, NORTH, DIR_SOUTH), NORTH);
    BOOST_CHECK_EQUAL(BoardUtils::PointInDir(*cb, NORTH, DIR_EAST), NORTH);
}

BOOST_AUTO_TEST_CASE(BoardUtils_ShiftBitset)
{
    BOOST_REQUIRE(MAX_WIDTH >= 8 && MAX_HEIGHT >= 8);
    ConstBoard* cb = &ConstBoard::Get(8, 8);
    bitset_t b1, b2;
    
    b1.set(HEX_CELL_A1);
    BOOST_CHECK(BoardUtils::ShiftBitset(*cb, b1, DIR_EAST, b2));
    BOOST_CHECK(b2.test(HEX_CELL_B1));
    
    BOOST_CHECK(!BoardUtils::ShiftBitset(*cb, b1, DIR_NORTH, b2));
    BOOST_CHECK(!BoardUtils::ShiftBitset(*cb, b1, DIR_WEST, b2));

    BOOST_CHECK(BoardUtils::ShiftBitset(*cb, b1, DIR_SOUTH, b2));
    BOOST_CHECK(b2.test(HEX_CELL_A2));

}

BOOST_AUTO_TEST_CASE(BoardUtil_RandomEmptyCell)
{
    HexPoint p;
    BOOST_REQUIRE(MAX_WIDTH >= 2 && MAX_HEIGHT >= 2);
    
    // test under normal conditions
    StoneBoard sb = StoneBoard(2, 2);
    
    p = BoardUtils::RandomEmptyCell(sb);
    BOOST_CHECK(sb.Const().IsCell(p));
    sb.StartNewGame();
    BOOST_CHECK(!sb.IsLegal(SWAP_PIECES));
    p = BoardUtils::RandomEmptyCell(sb);
    BOOST_CHECK(sb.Const().IsCell(p));
    sb.playMove(BLACK, HEX_CELL_A1);
    BOOST_CHECK(sb.IsLegal(SWAP_PIECES));
    sb.playMove(WHITE, HEX_CELL_A2);
    BOOST_CHECK(!sb.IsLegal(SWAP_PIECES));
    BOOST_CHECK_EQUAL(sb.GetPlayed().count(), 6u);
    BOOST_CHECK(!sb.IsEmpty(HEX_CELL_A1));
    BOOST_CHECK(!sb.IsEmpty(HEX_CELL_A2));
    
    p = BoardUtils::RandomEmptyCell(sb);
    BOOST_CHECK(sb.Const().IsCell(p));
    BOOST_CHECK(sb.IsEmpty(p));
    BOOST_CHECK(p != HEX_CELL_A1);
    BOOST_CHECK(p != HEX_CELL_A2);
    
    // test when one cell left
    sb = StoneBoard(1, 1);
    sb.StartNewGame();
    p = BoardUtils::RandomEmptyCell(sb);
    BOOST_CHECK_EQUAL(p, HEX_CELL_A1);
    
    // test when no cells left
    sb = StoneBoard(1, 1);
    sb.playMove(BLACK, HEX_CELL_A1);
    p = BoardUtils::RandomEmptyCell(sb);
    BOOST_CHECK_EQUAL(p, INVALID_POINT);
    
    // test when game has been resigned
    sb = StoneBoard(1, 1);
    sb.StartNewGame();
    sb.playMove(WHITE, RESIGN);
    BOOST_CHECK(!sb.IsLegal(HEX_CELL_A1));
    p = BoardUtils::RandomEmptyCell(sb);
    BOOST_CHECK_EQUAL(p, HEX_CELL_A1);
}

BOOST_AUTO_TEST_CASE(BoardUtil_Decompositions)
{
    ICEngine ice;
    VCBuilderParam param;
    HexBoard brd(7, 7, ice, param);
    
//  0xfc84158765cbbe20
//   a  b  c  d  e  f  g  
//  1\.  .  .  .  W  B  .\1
//   2\.  .  .  .  .  .  .\2
//    3\.  B  B  B  W  .  .\3
//     4\.  B  B  W  .  .  .\4
//      5\.  .  W  .  .  .  .\5
//       6\.  .  W  .  .  .  .\6
//        7\.  .  .  .  .  .  .\7
//           a  b  c  d  e  f  g'
    brd.GetState().playMove(BLACK, HEX_CELL_D3);
    brd.GetState().playMove(WHITE, HEX_CELL_E1);
    brd.GetState().playMove(BLACK, HEX_CELL_C4);
    brd.GetState().playMove(WHITE, HEX_CELL_D4);
    brd.GetState().playMove(BLACK, HEX_CELL_B4);
    brd.GetState().playMove(WHITE, HEX_CELL_C5);
    brd.GetState().playMove(BLACK, HEX_CELL_C3);
    brd.GetState().playMove(WHITE, HEX_CELL_C6);
    brd.GetState().playMove(BLACK, HEX_CELL_B3);
    brd.GetState().playMove(WHITE, HEX_CELL_E3);
    brd.GetState().playMove(BLACK, HEX_CELL_F1);

    // Find decomp between E1, B3, WEST, and NORTH. 
    brd.SetUseDecompositions(false);
    brd.ComputeAll(BLACK);
    brd.SetUseDecompositions(true);
    bitset_t capturedVC;
    BOOST_CHECK(BoardUtils::FindCombinatorialDecomposition(brd, BLACK, 
                                                           capturedVC));
    BOOST_CHECK(capturedVC.any());
}

BOOST_AUTO_TEST_CASE(BoardUtil_SplitDecompositions)
{
    ICEngine ice;
    VCBuilderParam param;
    HexBoard brd(7, 7, ice, param);
    
//  0xfc84158765cbbe20
//   a  b  c  d  e  f  g  
//  1\.  .  .  .  W  B  .\1
//   2\.  .  .  .  .  .  .\2
//    3\.  B  B  B  W  .  .\3
//     4\.  B  B  W  .  .  .\4
//      5\.  .  W  .  .  .  .\5
//       6\.  .  W  .  .  .  .\6
//        7\.  .  .  .  .  .  .\7
//           a  b  c  d  e  f  g'
    brd.GetState().playMove(BLACK, HEX_CELL_D3);
    brd.GetState().playMove(WHITE, HEX_CELL_E1);
    brd.GetState().playMove(BLACK, HEX_CELL_C4);
    brd.GetState().playMove(WHITE, HEX_CELL_D4);
    brd.GetState().playMove(BLACK, HEX_CELL_B4);
    brd.GetState().playMove(WHITE, HEX_CELL_C5);
    brd.GetState().playMove(BLACK, HEX_CELL_C3);
    brd.GetState().playMove(WHITE, HEX_CELL_C6);
    brd.GetState().playMove(BLACK, HEX_CELL_B3);
    brd.GetState().playMove(WHITE, HEX_CELL_E3);
    brd.GetState().playMove(BLACK, HEX_CELL_F1);

    // Find splitting decomp between NORTH, E3, SOUTH.
    brd.ComputeAll(WHITE);
    HexPoint group;
    BOOST_CHECK(BoardUtils::FindSplittingDecomposition(brd, WHITE, group));
    BOOST_CHECK_EQUAL(group, HEX_CELL_E3);
}

} // namespace

//---------------------------------------------------------------------------
