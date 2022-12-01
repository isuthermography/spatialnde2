
#ifndef SNDE_UNITS_HPP
#define SNDE_UNITS_HPP

#include <unordered_map>
#include <vector>

#include "snde/stringtools.hpp"

namespace snde {


  class UnitDef {
  public:
    std::string MeasName; // name of quantity measured by this unit
    std::string SingularName; // singular name of unit (e.g. meter)
    std::string PluralName; // plural name of unit (e.g. meters)
    std::vector<std::string> AbbrevList; // list of abbreviations
    size_t Index; // index of creation of this unit
    double PreferredPower; // Preferred power of 10; ... typically 0, but 3 for grams indicating usual use of kh
    bool SiPrefixFlag;

    UnitDef(std::string MeasName,std::string SingularName,std::string PluralName,std::vector<std::string> AbbrevList,size_t Index,double PreferredPower,bool SiPrefixFlag);
  };
  
  
  /* simplified definition found below 
  class SIPrefix {
  public:
    std::string Prefix;
    std::string Abbrev;
    int Power;

    SIPrefix(std::string Prefix,std::string Abbrev,int Power) :
      Prefix(Prefix),
      Abbrev(Abbrev),
      Power(Power)
    {

    }
  };

  static const std::vector<SIPrefix> SIPrefixes = {
						   { "Yotta","Y",24 },
						   { "Zetta", "Z", 21},
						   { "Exa", "E", 18},
						   {"Peta", "P", 15},
						   {"Tera", "T", 12},
						   {"Giga", "G", 9},
						   {"Mega", "M", 6},
						   {"kilo", "k", 3},
						   {"milli", "m", -3},
						   {"micro", "u", -6}, 
						   {"nano", "n", -9}, 
						   {"pico", "p", -12},
						   {"femto", "f", -15}, 
						   {"atto", "a", -18},
						   {"zepto", "z", -21}, 
						   {"yocto", "y", -24}
  };

  */
  
  class UnitFactor {
  public:
    std::shared_ptr<UnitDef> Unit;
    std::string NameOnly;
    double Power; // e.g. 1 for meters, 2 for meters^2, positive for numerator... COEFFICIENT WILL BE TAKEN TO THIS POWER TOO!
    double Coefficient; // Coefficient of these units. When normalized will be 1.0, except if there is a PreferredPower in which case this will be 10^PreferredPower

    UnitFactor(std::shared_ptr<UnitDef> Unit,std::string NameOnly, double Power, double Coefficient);
    

    friend int Compare(const UnitFactor &FactorA, const UnitFactor &FactorB);
    
    friend bool operator==(const UnitFactor &lhs, const UnitFactor &rhs);
    friend bool operator!=(const UnitFactor &lhs, const UnitFactor &rhs);
    friend bool operator<(const UnitFactor &lhs, const UnitFactor &rhs);
    friend bool operator>(const UnitFactor &lhs, const UnitFactor &rhs);
    friend bool operator<=(const UnitFactor &lhs, const UnitFactor &rhs);
    friend bool operator>=(const UnitFactor &lhs, const UnitFactor &rhs);
  };


  std::tuple<std::vector<UnitFactor>,double> sortunits(std::vector<UnitFactor> Factors);
  
  class UnitDB {
  public:
    std::unordered_map<std::string,std::shared_ptr<UnitDef>> UnitDict;
    std::unordered_map<std::string,std::shared_ptr<UnitDef>> MeasDict;
    
    UnitDB(std::vector<std::tuple<std::string,std::string,std::string,std::vector<std::string>,double,bool>> UnitDefs); // Unitdefs tuple consists of members of Class UnitDef, except for Index
    
  };
  
  
  

  std::string FactorName(const UnitFactor &Factor,bool longflag,bool pluralflag);
 
  
  std::shared_ptr<std::string> IsSiPrefix(double Power,bool longflag);
  
  std::tuple<bool,std::string,int> HasSIPrefix(std::string Name);


  

  
  class units {
  public:

    std::vector<UnitFactor> Factors;
    double Coefficient;

    units();
    units(std::vector<UnitFactor> Factors,double Coefficient);
    
    friend units operator*(const units& lhs,const units& rhs);
    friend units operator*(const units& lhs,double rhs);
    units& operator*=(const units& rhs);
    units& operator*=(double rhs);
    units power(double powerexp);
    friend units operator/(const units& lhs,const units& rhs);
    friend units operator/(const units& lhs,double rhs);
    units& operator/=(const units& rhs);
    units& operator/=(double rhs);
    units AddUnitFactor(std::string FactorName); // Note that unlike previous implementations, this returns a new units object 
    units simplify() const;
    std::string print(bool longflag=true);
    static std::tuple<std::string,units> parseunitpower_right(std::string unitstr);
    static std::tuple<std::string,units> parseunits_right(std::string unitstr);
    static units parseunits(std::string unitstr);
    static double comparerawunits(const units & CombA,const units &CombB);
    static bool compareunits(const units &comba, const units &combb);
  };

  class Equivalence {
  public:
    std::shared_ptr<units> ToReplace;
    std::shared_ptr<units> ReplaceWith;

    Equivalence(std::shared_ptr<units> ToReplace, std::shared_ptr<units> ReplaceWith);
    
  };

}

#endif // SNDE_UNITS_HPP
